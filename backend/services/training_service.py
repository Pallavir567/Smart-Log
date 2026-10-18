import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Built-in dataset loading via v2 pipeline (v1 had missing imports — fixed in v2)
from backend.training.train_ait_lds_v2 import load_dataset as builtin_load_dataset

from backend.dataset.loader import DatasetLoader
from backend.dataset.validator import DatasetValidator
from backend.preprocessing.pipeline import PreprocessingBundle
from backend.models.random_forest.rf_classifier import RandomForestModelWrapper
from backend.models.xgboost.xgb_classifier import XGBoostModelWrapper
from backend.models.registry import ModelRegistry
from backend.evaluation.metrics import EvaluationMetrics
from backend.services.classification_policy import RiskClassificationPolicy
from backend.database.db import query_one, execute_write

class TrainingService:
    """Orchestrates dataset validation, preprocessing, model training, and evaluation."""

    @classmethod
    def train_models(
        cls,
        dataset_id: str,
        algorithms: list = None,
        rf_params: Dict[str, Any] = None,
        xgb_params: Dict[str, Any] = None,
        split_config: Dict[str, Any] = None,
        behavioral_mode: bool = False
    ) -> Dict[str, Any]:
        """Train Random Forest and/or XGBoost on approved configured dataset.
        
        Args:
            dataset_id: Identifier of the dataset
            algorithms: List of algorithms to train ('random_forest', 'xgboost')
            rf_params: Hyperparameters for Random Forest
            xgb_params: Hyperparameters for XGBoost
            split_config: Custom split ratios
            
        Returns:
            Dict containing training results, metrics, and registered model IDs.
        """
        algorithms = algorithms or ["random_forest", "xgboost"]
        rf_params = rf_params or {}
        xgb_params = xgb_params or {}

        # 1. Fetch dataset record
        dataset_row = query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if not dataset_row:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        # 2. Fetch dataset config
        config_row = query_one("SELECT * FROM dataset_configs WHERE dataset_id = ?", (dataset_id,))
        if not config_row:
            raise ValueError(f"Dataset '{dataset_id}' has not been configured. Configure target labels and features first.")

        # Parse config
        feature_cols = json.loads(config_row["feature_columns_json"])
        config_dict = {
            "target_column": config_row["target_column"],
            "normal_value": config_row["normal_value"],
            "anomaly_value": config_row["anomaly_value"],
            "feature_columns": feature_cols,
            "train_ratio": float(split_config.get("train_ratio", config_row["train_ratio"])) if split_config else float(config_row["train_ratio"]),
            "val_ratio": float(split_config.get("val_ratio", config_row["val_ratio"])) if split_config else float(config_row["val_ratio"]),
            "test_ratio": float(split_config.get("test_ratio", config_row["test_ratio"])) if split_config else float(config_row["test_ratio"]),
            "stratified": bool(config_row["stratified"])
        }

        # 3. Load dataset
        if dataset_row["file_path"]:
            df, _ = DatasetLoader.load(dataset_row["file_path"])
        else:
            # Built-in dataset: use the built-in loader by dataset name (strip 'builtin_' prefix)
            builtin_name = dataset_row["id"].replace("builtin_", "")
            df, _ = builtin_load_dataset(builtin_name)

        # 4. Validate configuration
        is_valid, errors, warnings = DatasetValidator.validate_configuration(df, config_dict)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {'; '.join(errors)}")

        # 5. Filter unmapped records & map target
        target_col = config_dict["target_column"]
        norm_val = str(config_dict["normal_value"])
        anom_val = str(config_dict["anomaly_value"])

        # Drop rows where target is missing
        df_clean = df.dropna(subset=[target_col]).copy()
        df_clean["_str_target"] = df_clean[target_col].astype(str)
        # Keep only normal and anomaly rows
        df_clean = df_clean[df_clean["_str_target"].isin([norm_val, anom_val])].copy()

        # Map labels: normal -> 0, anomaly -> 1
        y = (df_clean["_str_target"] == anom_val).astype(int).values
        X = df_clean[feature_cols].copy()

        # 6. Train/Val/Test Split
        test_size = config_dict["test_ratio"]
        val_size = config_dict["val_ratio"]
        train_size = config_dict["train_ratio"]

        # Stratified if classes permit
        strat = y if (config_dict["stratified"] and len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2) else None

        # First split into train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=42
        )

        # Then split train_val into train and val
        val_relative_ratio = val_size / (train_size + val_size) if (train_size + val_size) > 0 else 0.15
        strat_tv = y_train_val if (config_dict["stratified"] and len(np.unique(y_train_val)) > 1 and np.min(np.bincount(y_train_val)) >= 2) else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative_ratio, stratify=strat_tv, random_state=42
        )

        # 7. Fit Preprocessor strictly on training data
        # 7a. Behavioral feature extraction (if enabled and timestamp column exists)
        behavioral_extractor = None
        if behavioral_mode:
            ts_col = config_row.get("timestamp_column")
            host_available = "hostname" in df_clean.columns
            if ts_col and ts_col in df_clean.columns:
                try:
                    from backend.feature_engineering.behavioral_features import (
                        BehavioralWindowExtractor, EventNormalizer, BEHAVIORAL_FEATURE_NAMES,
                    )
                    import logging as _beh_log
                    _logger = _beh_log.getLogger("training_service")
                    _logger.info("Behavioral mode: extracting 13 time-window features...")
                    normalizer = EventNormalizer()
                    behavioral_extractor = BehavioralWindowExtractor(window_minutes=5, grouping_keys=("hostname",))

                    def _make_norm(row):
                        return normalizer.normalize({
                            "hostname": str(row.get("hostname", "localhost")),
                            "timestamp": str(row.get(ts_col, "")),
                            "service": str(row.get("service", "unknown")),
                            "message": str(row.get("raw_log", row.get("message", ""))),
                            "_features": {k: row.get(k, 0) for k in feature_cols},
                        }, log_type="syslog")

                    tr_norm = [_make_norm(X_train.iloc[i]) for i in range(len(X_train))]
                    behavioral_extractor.fit_baseline(tr_norm)
                    tr_beh = behavioral_extractor.extract_batch(tr_norm)
                    val_norm = [_make_norm(X_val.iloc[i]) for i in range(len(X_val))]
                    val_beh = behavioral_extractor.extract_batch(val_norm)
                    test_norm = [_make_norm(X_test.iloc[i]) for i in range(len(X_test))]
                    test_beh = behavioral_extractor.extract_batch(test_norm)

                    X_train = X_train.copy().reset_index(drop=True)
                    X_val = X_val.copy().reset_index(drop=True)
                    X_test = X_test.copy().reset_index(drop=True)
                    for feat in BEHAVIORAL_FEATURE_NAMES:
                        X_train[feat] = [b.get(feat, 0) for b in tr_beh]
                        X_val[feat]   = [b.get(feat, 0) for b in val_beh]
                        X_test[feat]  = [b.get(feat, 0) for b in test_beh]
                    feature_cols = feature_cols + BEHAVIORAL_FEATURE_NAMES
                    _logger.info("Behavioral features added. Total: %d features.", len(feature_cols))
                except Exception as beh_err:
                    import logging as _beh_log
                    _beh_log.getLogger("training_service").warning(
                        "Behavioral feature extraction failed, falling back to message-level: %s", beh_err
                    )
                    behavioral_extractor = None

        bundle = PreprocessingBundle.infer_and_create(X_train, feature_cols)
        X_train_trans = bundle.fit_transform(X_train)
        X_val_trans = bundle.transform(X_val)
        X_test_trans = bundle.transform(X_test)

        dataset_meta = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_row["name"],
            "label_mapping": {"normal": norm_val, "anomaly": anom_val}
        }

        trained_results = {}

        # 8. Train Random Forest
        if "random_forest" in algorithms:
            rf_id = f"rf_{uuid.uuid4().hex[:8]}"
            rf_model = RandomForestModelWrapper(model_id=rf_id, hyperparameters=rf_params)
            rf_model.feature_names = bundle.transformed_feature_names

            start_t = time.time()
            rf_model.fit(X_train_trans, y_train)
            duration = round(time.time() - start_t, 3)

            rf_preds = rf_model.predict(X_test_trans)
            rf_probs = rf_model.predict_proba(X_test_trans)
            rf_metrics = EvaluationMetrics.evaluate(y_test, rf_preds, rf_probs)
            rf_metrics["training_duration_seconds"] = duration
            rf_metrics["test_samples"] = len(y_test)
            rf_metrics["train_samples"] = len(y_train)
            rf_metrics["val_samples"] = len(y_val)

            # Validate 3-level classification policy on validation split
            try:
                rf_val_probs = rf_model.predict_proba(X_val_trans)
                rf_val_prob_anom = rf_val_probs[:, 1] if rf_val_probs.shape[1] > 1 else rf_val_probs[:, 0]
                policy = RiskClassificationPolicy()
                rf_metrics["classification_policy"] = policy.validate_on_validation_set(
                    y_val=list(y_val),
                    y_probs=[float(p) for p in rf_val_prob_anom]
                )
            except Exception as e:
                rf_metrics["classification_policy"] = {"validated": False, "error": str(e)}

            saved_rf = ModelRegistry.save_model(
                model_wrapper=rf_model,
                preprocessor=bundle,
                metrics=rf_metrics,
                dataset_meta=dataset_meta,
                version=f"1.0.{int(time.time()) % 1000}",
                behavioral_extractor=behavioral_extractor
            )
            trained_results["random_forest"] = {
                "model_id": rf_id,
                "name": rf_model.name,
                "metrics": rf_metrics,
                "artifact": saved_rf
            }

        # 9. Train XGBoost
        if "xgboost" in algorithms:
            xgb_id = f"xgb_{uuid.uuid4().hex[:8]}"
            xgb_model = XGBoostModelWrapper(model_id=xgb_id, hyperparameters=xgb_params)
            xgb_model.feature_names = bundle.transformed_feature_names

            start_t = time.time()
            xgb_model.fit(X_train_trans, y_train, X_val_trans, y_val)
            duration = round(time.time() - start_t, 3)

            xgb_preds = xgb_model.predict(X_test_trans)
            xgb_probs = xgb_model.predict_proba(X_test_trans)
            xgb_metrics = EvaluationMetrics.evaluate(y_test, xgb_preds, xgb_probs)
            xgb_metrics["training_duration_seconds"] = duration
            xgb_metrics["test_samples"] = len(y_test)
            xgb_metrics["train_samples"] = len(y_train)
            xgb_metrics["val_samples"] = len(y_val)

            # Validate 3-level classification policy on validation split
            try:
                xgb_val_probs = xgb_model.predict_proba(X_val_trans)
                xgb_val_prob_anom = xgb_val_probs[:, 1] if xgb_val_probs.shape[1] > 1 else xgb_val_probs[:, 0]
                policy = RiskClassificationPolicy()
                xgb_metrics["classification_policy"] = policy.validate_on_validation_set(
                    y_val=list(y_val),
                    y_probs=[float(p) for p in xgb_val_prob_anom]
                )
            except Exception as e:
                xgb_metrics["classification_policy"] = {"validated": False, "error": str(e)}

            saved_xgb = ModelRegistry.save_model(
                model_wrapper=xgb_model,
                preprocessor=bundle,
                metrics=xgb_metrics,
                dataset_meta=dataset_meta,
                version=f"1.0.{int(time.time()) % 1000}",
                behavioral_extractor=behavioral_extractor
            )
            trained_results["xgboost"] = {
                "model_id": xgb_id,
                "name": xgb_model.name,
                "metrics": xgb_metrics,
                "artifact": saved_xgb
            }

        return {
            "success": True,
            "dataset_id": dataset_id,
            "dataset_name": dataset_row["name"],
            "feature_mode": "behavioral" if behavioral_extractor is not None else "message_level",
            "split_summary": {
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "total_valid": len(df_clean)
            },
            "models": trained_results,
            "warnings": warnings
        }
