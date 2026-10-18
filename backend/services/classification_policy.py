import math
from typing import Dict, Any, List, Optional, Tuple

# Technical to plain-English feature translations
FEATURE_TRANSLATIONS: Dict[str, str] = {
    "message_length": "Log message character size",
    "token_count": "Word count in log message",
    "digit_ratio": "Proportion of numeric characters",
    "entropy": "Text randomness and complexity (entropy)",
    "has_error": "Presence of system error or failure keywords",
    "has_warning": "Warning severity indicator in log",
    "has_auth": "Authentication or credential activity",
    "has_failed_auth": "Failed login or password failure attempts",
    "has_denied": "Access denied or unauthorized indicator",
    "has_ip": "Embedded IP network address",
    "service": "Originating service or application name",
    "failed_login_attempts": "Number of failed login attempts",
    "event_frequency": "Rate of events in time window",
    "error_count": "Observed system error count",
    "unknown_events": "Count of unrecognized event types",
    "sequence_length": "Observed event sequence length"
}

class RiskClassificationPolicy:
    """Configurable three-level risk classification policy with transparent decision layers and plain-English explainability."""

    # Default calibrated threshold values
    DEFAULT_THRESHOLD_NORMAL: float = 0.30
    DEFAULT_THRESHOLD_ANOMALY: float = 0.70

    def __init__(
        self,
        threshold_normal: float = DEFAULT_THRESHOLD_NORMAL,
        threshold_anomaly: float = DEFAULT_THRESHOLD_ANOMALY,
        policy_name: str = "Configured Probability Threshold Policy"
    ):
        if not (0.0 <= threshold_normal < threshold_anomaly <= 1.0):
            raise ValueError(
                f"Invalid policy thresholds: 0.0 <= threshold_normal ({threshold_normal}) "
                f"< threshold_anomaly ({threshold_anomaly}) <= 1.0 is required."
            )
        self.threshold_normal = round(float(threshold_normal), 4)
        self.threshold_anomaly = round(float(threshold_anomaly), 4)
        self.policy_name = policy_name

    def classify(
        self,
        probability: Optional[float],
        original_prediction: Optional[int] = None,
        explicit_state: Optional[str] = None
    ) -> str:
        """Determine risk category from model probability or explicit lifecycle state.
        
        Returns:
            'Normal', 'Medium', 'Anomalous', 'Pending Analysis', 'Not Evaluated',
            'Classification Error', or 'Unclassified'
        """
        if explicit_state:
            state_lower = explicit_state.lower().strip()
            if state_lower in ("pending", "pending_analysis", "pending analysis"):
                return "Pending Analysis"
            if state_lower in ("not_evaluated", "not evaluated", "incompatible"):
                return "Not Evaluated"
            if state_lower in ("error", "classification_error", "classification error"):
                return "Classification Error"
            if state_lower in ("normal", "medium", "anomalous", "unclassified"):
                return explicit_state.title()

        if probability is None or not isinstance(probability, (int, float)) or math.isnan(probability):
            return "Unclassified"

        prob = float(probability)
        if prob < self.threshold_normal:
            return "Normal"
        elif prob < self.threshold_anomaly:
            return "Medium"
        else:
            return "Anomalous"

    def calculate_confidence(self, probability: Optional[float]) -> Optional[float]:
        """Calculate decision confidence score only when supported.
        
        Returns confidence in [0.0, 1.0], or None if unsupported.
        For binary probabilities, confidence represents distance from ambiguous center (0.5).
        """
        if probability is None or not isinstance(probability, (int, float)) or math.isnan(probability):
            return None
        prob = float(probability)
        conf = abs(prob - 0.5) * 2.0
        return round(conf, 4)

    def build_explanation(
        self,
        record: Dict[str, Any],
        features: Dict[str, Any],
        category: str,
        probability: Optional[float],
        confidence: Optional[float],
        original_prediction: Optional[int],
        shap_data: Optional[Dict[str, Any]] = None,
        model_meta: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Construct structured explanation separating observed facts, model explanation,
        security interpretations, and plain-English non-expert translations.
        """
        raw_msg = str(record.get("message") or record.get("raw") or "")
        app_name = str(record.get("provider") or record.get("service") or record.get("process_name") or "System")
        hostname = str(record.get("hostname") or record.get("computer") or "localhost")
        username = record.get("username") or record.get("user_id") or features.get("username")
        action = record.get("action") or record.get("event_type")

        # 1. Observed Facts
        observed_facts = {
            "timestamp": record.get("timestamp") or "N/A",
            "hostname": hostname,
            "application_name": app_name,
            "channel": record.get("channel") or "System",
            "event_id": record.get("event_id"),
            "record_id": record.get("record_id"),
            "action": action,
            "log_level": self._decode_log_level(record.get("severity")),
            "raw_severity_code": record.get("severity"),
            "message_length": len(raw_msg),
            "entropy": features.get("entropy", 0.0),
            "token_count": features.get("token_count", 0),
            "detected_source_ip": record.get("source_ip") or features.get("source_ip"),
            "detected_user": username,
            "has_error_flag": bool(features.get("has_error", 0)),
            "has_auth_flag": bool(features.get("has_auth", 0)),
            "has_failed_auth_flag": bool(features.get("has_failed_auth", 0)),
            "has_denied_flag": bool(features.get("has_denied", 0)),
            "has_warning_flag": bool(features.get("has_warning", 0)),
            "has_suspicious_context": bool(record.get("has_suspicious_context", False))
        }

        # 2. Extract SHAP drivers and translate to plain English
        top_positive_shap = []
        top_negative_shap = []
        plain_contributing_factors = []

        shap_available = bool(shap_data and shap_data.get("available") and shap_data.get("contributions"))

        if shap_available:
            for c in shap_data["contributions"]:
                val = c.get("contribution", 0)
                feat_raw_name = c.get("feature", "")
                # Clean one-hot prefixes if present (e.g. cat__service_sshd -> service)
                clean_name = feat_raw_name
                if "cat__" in clean_name:
                    clean_name = clean_name.split("cat__")[-1]
                if "__" in clean_name:
                    clean_name = clean_name.split("__")[-1]
                if clean_name.startswith("service_"):
                    clean_name = "service"

                translated_title = FEATURE_TRANSLATIONS.get(clean_name, clean_name.replace("_", " ").title())
                
                is_pushing_anomaly = (c.get("direction") == "anomaly" or val > 0)
                if is_pushing_anomaly:
                    top_positive_shap.append(c)
                else:
                    top_negative_shap.append(c)

                # Format human influence
                abs_val = abs(val)
                if abs_val >= 0.15:
                    strength = "High influence"
                elif abs_val >= 0.05:
                    strength = "Moderate influence"
                else:
                    strength = "Minor influence"

                plain_contributing_factors.append({
                    "factor": translated_title,
                    "technical_feature": feat_raw_name,
                    "observed_value": str(c.get("feature_value", "N/A")),
                    "influence": "Supported Anomaly prediction" if is_pushing_anomaly else "Supported Normal prediction",
                    "impact_strength": strength,
                    "shap_contribution": round(val, 4)
                })

        prob_pct = f"{round(probability * 100, 1)}%" if probability is not None else "N/A"
        norm_thresh_pct = f"{round(self.threshold_normal * 100, 1)}%"
        anom_thresh_pct = f"{round(self.threshold_anomaly * 100, 1)}%"

        # 3. Dynamic Plain-English Explanation Generation
        has_auth_fail = bool(features.get("has_failed_auth") or "failed password" in raw_msg.lower() or "logon failure" in raw_msg.lower())
        has_suspicious_proc = bool(record.get("has_suspicious_context") or "suspicious" in raw_msg.lower())
        is_app_lifecycle = bool(action in ("APPLICATION_OPENED", "APPLICATION_CLOSED") or record.get("is_target_app"))

        if category == "Normal":
            if is_app_lifecycle:
                why = (
                    f"Classified as Normal because the application activity for '{app_name}' matches the system's usual behavior. "
                    f"The event parameters, user context ('{username or 'Standard'}'), and error count are within expected operational bounds, "
                    f"and no suspicious indicators were detected (model anomaly probability: {prob_pct}, below {norm_thresh_pct} threshold)."
                )
            else:
                why = (
                    f"Classified as Normal because the event characteristics conform to the system's learned operational baseline. "
                    f"The model predicted an anomaly probability of {prob_pct}, which is strictly below the normal threshold of {norm_thresh_pct}. "
                    f"No strong anomaly indicators were detected."
                )
            how_influenced = (
                "Baseline operational feature values, normal text characteristics, and absence of failure keywords "
                "consistently pushed the model score towards a Normal prediction."
            )
            recommended_action = "No immediate action required. Event conforms to normal operational baseline."
            fallback_msg = None

        elif category == "Medium":
            why = (
                f"Classified as Medium Risk because the event exhibits intermediate characteristics with an anomaly probability of {prob_pct}, "
                f"falling in the uncertainty range between Normal ({norm_thresh_pct}) and Anomalous ({anom_thresh_pct}). "
                f"Features exhibit mild deviation from the training baseline without meeting definitive anomaly criteria."
            )
            how_influenced = (
                "Feature contributions are mixed: some indicators support baseline behavior while other factors "
                "(such as warning level or atypical message entropy) pushed the probability upwards into the uncertainty band."
            )
            recommended_action = (
                f"Monitor subsequent activity from application '{app_name}' or host '{hostname}'. "
                f"Correlate with related events if further deviations occur."
            )
            fallback_msg = None

        elif category == "Anomalous":
            if has_auth_fail:
                why = (
                    f"Classified as Anomaly because the failed login or authentication indicator is elevated compared with "
                    f"the patterns learned by the model (model anomaly probability: {prob_pct} ≥ {anom_thresh_pct} threshold). "
                    f"The presence of credential failure tokens contributed strongly to the anomaly prediction."
                )
                recommended_action = (
                    f"Review the related authentication events and source identity for '{hostname}' "
                    f"to determine whether this indicates a possible brute-force attempt or credential attack."
                )
            elif has_suspicious_proc:
                why = (
                    f"This process activity was flagged as Anomaly because its execution parameters, command-line arguments, "
                    f"or executable location differ significantly from normal application patterns (anomaly probability: {prob_pct} ≥ {anom_thresh_pct}). "
                    f"This is a potential anomaly, not confirmation of malicious activity."
                )
                recommended_action = (
                    f"Review the process execution details, parent process, and command-line arguments for '{app_name}' on '{hostname}'."
                )
            else:
                why = (
                    f"Classified as Anomaly because the event significantly deviates from baseline operational models "
                    f"with an anomaly probability of {prob_pct} (≥ {anom_thresh_pct} threshold). "
                    f"Statistical outliers in message length, entropy, or service patterns drove the elevated score."
                )
                recommended_action = (
                    f"Investigate host logs, originating service '{app_name}', and related session activity on '{hostname}'. "
                    f"Note: Anomaly classification denotes statistical deviation from baseline, not confirmed compromise."
                )
            how_influenced = (
                "Strong feature contributions pushing towards Anomaly dominated the model decision boundary, "
                "elevating the prediction score past the high-risk threshold."
            )
            fallback_msg = None

        elif category == "Pending Analysis":
            why = "This event has been ingested into the system and is currently pending ML evaluation."
            how_influenced = "Inference has not yet been executed on this event record."
            recommended_action = "Execute batch or live inference to obtain model evaluation and SHAP attributions."
            fallback_msg = "Event is pending evaluation."

        elif category == "Not Evaluated":
            why = (
                f"This event was not evaluated because no active model with a compatible feature schema is available, "
                f"or incoming event features could not be mapped to the loaded classifier ({error_message or 'schema mismatch'})."
            )
            how_influenced = "Feature values were parsed successfully, but could not be transformed against the active model schema."
            recommended_action = "Train and activate a compatible model in the Model Registry supporting this event schema."
            fallback_msg = "The model was not evaluated because a compatible model artifact is unavailable."

        elif category == "Classification Error":
            why = f"An unexpected error occurred during model inference: {error_message or 'internal exception'}."
            how_influenced = "The prediction pipeline could not complete due to an execution failure."
            recommended_action = "Review application logs and model artifact integrity to diagnose the runtime error."
            fallback_msg = "Classification error: reliable feature-level explanation is unavailable."

        else:  # Legacy Unclassified fallback
            why = (
                "Unclassified event: A reliable model decision could not be made. The active classifier either did not output "
                "class probabilities, or incoming features could not be evaluated against the configured policy."
            )
            how_influenced = "Classification cannot be performed without valid model probability outputs."
            recommended_action = "Review model configuration and ensure an active classifier is loaded."
            fallback_msg = "The model produced a prediction, but a reliable feature-level explanation is unavailable."

        # If SHAP is missing for an evaluated prediction, provide honest fallback message
        if not shap_available and category in ("Normal", "Medium", "Anomalous"):
            fallback_msg = "The model produced a prediction, but a reliable feature-level explanation is unavailable."

        # Plain English structured summary
        summary = why

        model_explanation = {
            "model_name": model_meta.get("name", "N/A") if model_meta else "N/A",
            "algorithm": model_meta.get("algorithm", "N/A") if model_meta else "N/A",
            "model_version": model_meta.get("version", "N/A") if model_meta else "N/A",
            "original_prediction": original_prediction,
            "risk_score_probability": probability,
            "confidence_score": confidence,
            "policy_threshold_normal": self.threshold_normal,
            "policy_threshold_anomaly": self.threshold_anomaly,
            "decision_rationale": why,
            "uncertainty_note": (
                "Classification exhibits moderate uncertainty." if category == "Medium"
                else "Model confidence is high for operational conformity." if category == "Normal"
                else "Model confidence is high for anomaly classification." if category == "Anomalous"
                else "Classification cannot be performed without valid model probability outputs." if category == "Unclassified"
                else "Evaluation status: " + category
            ),
            "top_anomaly_drivers": top_positive_shap[:5],
            "top_normal_drivers": top_negative_shap[:5]
        }

        # Operational security interpretation
        if category == "Normal":
            sec_interpretation = (
                f"Observed activity from '{app_name}' matches typical operational baseline patterns. "
                f"No indicators of compromise, anomalous execution behaviors, or privilege escalation patterns were detected."
            )
        elif category == "Medium":
            sec_interpretation = (
                f"Observed event from '{app_name}' exhibits atypical characteristics that warrant operational review, "
                f"but does not meet definitive anomaly criteria. Recommended operational action: correlate with subsequent session events."
            )
        elif category == "Anomalous":
            sec_interpretation = (
                f"Observed event significantly deviates from baseline operational models, exhibiting elevated risk indicators. "
                f"Recommended operational action: investigate host '{hostname}', source IP, and related process activity. "
                f"Note: Anomaly classification indicates statistical deviation from training baseline, not a confirmed attack."
            )
        else:
            sec_interpretation = (
                "Reliable risk assessment cannot be determined from available model telemetry. "
                "Event is retained in audit log for administrative review."
            )

        plain_english = {
            "classification": category,
            "why": why,
            "important_contributing_factors": plain_contributing_factors[:6],
            "how_factors_influenced": how_influenced,
            "recommended_action": recommended_action,
            "fallback_message": fallback_msg
        }

        return {
            "summary": summary,
            "observed_facts": observed_facts,
            "model_explanation": model_explanation,
            "security_interpretation": sec_interpretation,
            "plain_english": plain_english
        }

    @staticmethod
    def _decode_log_level(sev: Optional[Any]) -> str:
        """Decode numerical severity code into standard syslog severity string."""
        if sev is None:
            return "INFO"
        try:
            s = int(sev)
            mapping = {
                0: "EMERGENCY",
                1: "ALERT",
                2: "CRITICAL",
                3: "ERROR",
                4: "WARNING",
                5: "NOTICE",
                6: "INFO",
                7: "DEBUG"
            }
            return mapping.get(s, f"LEVEL_{s}")
        except Exception:
            return str(sev).upper()

    def validate_on_validation_set(
        self,
        y_val: List[int],
        y_probs: List[float]
    ) -> Dict[str, Any]:
        """Validate threshold policy on held-out validation set and evaluate separation."""
        total = len(y_val)
        if total == 0:
            return {
                "validated": False,
                "reason": "Validation set is empty"
            }

        normal_count = 0
        medium_count = 0
        anomalous_count = 0

        normal_true_anomalies = 0
        anomalous_true_anomalies = 0

        for true_label, prob in zip(y_val, y_probs):
            cat = self.classify(prob)
            if cat == "Normal":
                normal_count += 1
                if true_label == 1:
                    normal_true_anomalies += 1
            elif cat == "Medium":
                medium_count += 1
            elif cat == "Anomalous":
                anomalous_count += 1
                if true_label == 1:
                    anomalous_true_anomalies += 1

        return {
            "validated": True,
            "threshold_normal": self.threshold_normal,
            "threshold_anomaly": self.threshold_anomaly,
            "total_validation_samples": total,
            "distribution": {
                "normal_count": normal_count,
                "normal_percent": round((normal_count / total) * 100, 2),
                "medium_count": medium_count,
                "medium_percent": round((medium_count / total) * 100, 2),
                "anomalous_count": anomalous_count,
                "anomalous_percent": round((anomalous_count / total) * 100, 2)
            },
            "anomalous_precision": round(anomalous_true_anomalies / anomalous_count, 4) if anomalous_count > 0 else 0.0,
            "normal_false_negative_rate": round(normal_true_anomalies / normal_count, 4) if normal_count > 0 else 0.0
        }
