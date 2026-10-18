import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap,
  Play,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Terminal,
  Activity,
  Layers,
  Info,
  Clock,
  Sparkles,
  Server,
  FileText,
  Copy,
  Check,
  ShieldAlert,
  ShieldCheck,
  HelpCircle,
  Hash
} from 'lucide-react';
import { api } from '../services/api';
import { PredictionRecord, SimulationPreset, SimulationHistoryItem, ShapContribution } from '../types';
import { PageHeader } from '../components/common/PageHeader';
import { CyberCard } from '../components/common/CyberCard';
import { StatusBadge } from '../components/common/StatusBadge';
import { EmptyState } from '../components/common/EmptyState';

// 6 Core Pre-Calibrated Scenarios
const PRESET_SCENARIOS: SimulationPreset[] = [
  {
    id: 'win_logon_4625',
    name: 'Windows Failed Logon — Event ID 4625',
    category: 'Authentication & Credential Access',
    threatType: 'Anomalous',
    expectedCategory: 'Anomalous',
    description: 'Simulates an interactive logon failure logged by Windows Security Auditing (Status: 0xC000006D, SubStatus: 0xC000006A).',
    whyUseful: 'Tests extraction of Windows auditing fields, detection of bad passwords, and attribution of high-risk security events without altering OS accounts.',
    nature: 'Suspicious / Interactive Authentication Failure',
    defaultPayload: '2026-10-17T12:00:01 SahanaMayar Microsoft-Windows-Security-Auditing[4625]: An account failed to log on. Account Name: Administrator. Failure Reason: Unknown user name or bad password. Status: 0xC000006D Sub Status: 0xC000006A',
  },
  {
    id: 'ssh_invalid_user',
    name: 'SSH Invalid-User Login Attempt',
    category: 'Remote Access & Credential Harvesting',
    threatType: 'Anomalous',
    expectedCategory: 'Anomalous',
    description: 'Simulates a Linux SSH daemon log recording a password authentication attempt for an unrecognized username from an external IPv4 address.',
    whyUseful: 'Validates RFC 3164 BSD syslog parsing, regex user/IP extraction, and SSH auth-failure lexical weighting in the active model.',
    nature: 'Suspicious / Non-Existent Account Probe',
    defaultPayload: 'Oct 17 12:00:02 server sshd[1236]: Failed password for invalid user admin from 10.0.0.99 port 59123 ssh2',
  },
  {
    id: 'powershell_evasion',
    name: 'Suspicious PowerShell Execution',
    category: 'Evasion & Malicious Process Execution',
    threatType: 'Anomalous',
    expectedCategory: 'Anomalous',
    description: 'Simulates an application launch log for PowerShell executing with evasion arguments (-nop -w hidden -enc) from a Temp path.',
    whyUseful: 'Tests endpoint process tracking, high Shannon text entropy calculation, and process-rule anomaly overrides.',
    nature: 'Suspicious / Command-Line Evasion Context',
    defaultPayload: '2026-10-17T12:00:09 SAHANAMAYAR powershell.exe[3344]: [APPLICATION_OPENED] Windows PowerShell opened by user SYSTEM. Executable: C:\\Windows\\Temp\\powershell.exe | Command Line: powershell.exe -nop -w hidden -enc SIMULATED_PAYLOAD [Simulation: inert test string]',
  },
  {
    id: 'web_dir_traversal',
    name: 'Web Directory Traversal Attempt',
    category: 'Web Application Attack / Path Traversal',
    threatType: 'Anomalous',
    expectedCategory: 'Anomalous',
    description: 'Simulates an Nginx web server access log capturing a directory traversal probe (/../../etc/passwd) from a web vulnerability scanner.',
    whyUseful: 'Evaluates web service anomaly detection based on statistical text structure, character ratios, and scanner signatures without auth keywords.',
    nature: 'Suspicious / Path Traversal Probe',
    defaultPayload: 'Oct 17 12:00:07 server nginx: 10.0.0.150 - - [17/Oct/2026:12:00:07 +0000] "GET /../../etc/passwd HTTP/1.1" 400 166 "dirb/2.22"',
  },
  {
    id: 'ssh_ambiguous',
    name: 'Ambiguous SSH Public-Key Login',
    category: 'Boundary / Mixed Security Signals',
    threatType: 'Ambiguous',
    expectedCategory: 'Medium',
    description: 'Simulates a valid SSH public-key authentication session opened from an internal IP address.',
    whyUseful: "Demonstrates Smart-Log's three-level classification policy and the [0.30, 0.70) uncertainty band where mixed indicators produce a Medium risk score.",
    nature: 'Boundary / Intermediate Model Uncertainty',
    defaultPayload: 'Oct 17 12:00:00 server sshd[1234]: Accepted publickey for admin from 192.168.1.15 port 52412 ssh2',
  },
  {
    id: 'notepad_normal',
    name: 'Normal Notepad Application Launch',
    category: 'Benign Operational Baseline Control',
    threatType: 'Benign',
    expectedCategory: 'Normal',
    description: 'Simulates a routine administrative text editor launch from System32 with standard parameters.',
    whyUseful: 'Serves as a true-negative baseline control to verify that routine endpoint software activity correctly scores as Normal.',
    nature: 'Benign / Expected System Operation',
    defaultPayload: "2026-10-17T12:00:10 SAHANAMAYAR notepad.exe[4120]: [APPLICATION_OPENED] Notepad Text Editor ('notepad.exe') opened by user 'Admin' (PID: 4120). Executable: C:\\Windows\\System32\\notepad.exe",
  },
];

export const SimulationLabPage: React.FC = () => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('win_logon_4625');
  const [payload, setPayload] = useState<string>(PRESET_SCENARIOS[0].defaultPayload);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<PredictionRecord | null>(null);
  const [activeTab, setActiveTab] = useState<'explanation' | 'shap' | 'facts' | 'raw'>('explanation');
  const [copied, setCopied] = useState<boolean>(false);
  const [history, setHistory] = useState<SimulationHistoryItem[]>([]);

  const selectedPreset = PRESET_SCENARIOS.find((p) => p.id === selectedPresetId) || PRESET_SCENARIOS[0];

  const handleSelectPreset = (preset: SimulationPreset) => {
    setSelectedPresetId(preset.id);
    setPayload(preset.defaultPayload);
    setErrorMsg(null);
  };

  const handleResetPayload = () => {
    setPayload(selectedPreset.defaultPayload);
    setErrorMsg(null);
  };

  const handleCopyPayload = () => {
    navigator.clipboard.writeText(payload);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleInject = async () => {
    const trimmed = payload.trim();
    if (!trimmed) {
      setErrorMsg('Payload cannot be empty. Please provide a valid log string to evaluate.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    try {
      // Pass channel = 'Simulation' to tag the record and isolate it from live Windows telemetry
      const response = await api.predictLogs([trimmed], true, 'Simulation');

      if (!response.success || !response.results || response.results.length === 0) {
        throw new Error(response.error || 'The model did not return any prediction for this input.');
      }

      const rec: PredictionRecord = response.results[0];
      setCurrentResult(rec);

      // Append to session simulation history
      const historyItem: SimulationHistoryItem = {
        id: rec.id || `sim_${Date.now()}`,
        scenarioName: selectedPreset.name,
        category: selectedPreset.category,
        payloadSnippet: trimmed.length > 85 ? trimmed.substring(0, 85) + '...' : trimmed,
        fullPayload: trimmed,
        classification: rec.category || rec.classification || 'Unclassified',
        probability: rec.probability ?? null,
        confidence: rec.confidence ?? null,
        timestamp: new Date().toLocaleTimeString(),
        success: true,
        predictionRecord: rec,
      };

      setHistory((prev) => [historyItem, ...prev.slice(0, 19)]);
    } catch (err: any) {
      setErrorMsg(err.message || 'Inference pipeline encountered an unexpected error.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInspectHistory = (item: SimulationHistoryItem) => {
    if (item.predictionRecord) {
      setCurrentResult(item.predictionRecord);
      setPayload(item.fullPayload);
    }
  };

  return (
    <div className="space-y-6">
      {/* 1. Header and Page Banner */}
      <PageHeader
        title="Anomaly Simulation Lab"
        subtitle="Safely simulate security log events and evaluate Smart-Log's real detection pipeline in real time."
        badge="SIMULATION MODE"
        actions={
          <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/70 border border-cyan-500/40 text-cyan-300">
            <Zap className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>INERT TEST ENVIRONMENT</span>
          </div>
        }
      />

      {/* Behavioral Cold-Start Notice */}
      <div className="p-3.5 rounded-xl bg-violet-950/30 border border-violet-500/30 text-[11px] font-mono text-violet-300 flex items-start gap-2.5">
        <span className="mt-0.5 shrink-0">⚡</span>
        <span>
          <strong>Behavioral Feature Mode:</strong> Single-event simulation runs in{' '}
          <strong>cold-start mode</strong> — time-window behavioral features (auth fail rate, privilege count, etc.) are set to 0 with <code>beh_cold_start=1</code>. Multi-event behavioral patterns require batch file analysis.
        </span>
      </div>

      {/* Safety & Methodology Notice */}
      <div className="p-4 rounded-xl bg-[#09101F] border border-cyan-500/30 text-xs font-sans text-slate-300 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-lg">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-white tracking-wide">
              Stateless Single-Event Point-in-Time Evaluation
            </p>
            <p className="text-slate-400 leading-relaxed">
              Every injected scenario is an inert test string processed through Smart-Log&apos;s real parser, feature extractor, XGBoost classifier, and SHAP attribution engine. No live attacks or commands are executed. Note: Single events run in cold-start mode (<code>beh_cold_start=1</code>); multi-event sequence patterns and time-window behavioral aggregation require batch file analysis.
            </p>
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0E1A33] border border-[#1E3050] text-[11px] font-mono text-cyan-300">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>REAL PIPELINE ACTIVE</span>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-rose-950/70 border border-rose-500/60 text-rose-200 text-xs font-mono flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg(null)} className="text-rose-400 hover:text-white">
            &times;
          </button>
        </div>
      )}

      {/* 2. Scenario Selector Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Select Test Scenario Preset</span>
          </h3>
          <span className="text-xs text-slate-500 font-mono">6 Pre-Calibrated Scenarios</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {PRESET_SCENARIOS.map((scenario) => {
            const isSelected = scenario.id === selectedPresetId;
            const badgeColor =
              scenario.threatType === 'Anomalous'
                ? 'bg-rose-950/70 text-rose-300 border-rose-500/40'
                : scenario.threatType === 'Ambiguous'
                ? 'bg-amber-950/70 text-amber-300 border-amber-500/40'
                : 'bg-emerald-950/70 text-emerald-300 border-emerald-500/40';

            return (
              <CyberCard
                key={scenario.id}
                onClick={() => handleSelectPreset(scenario)}
                glow={isSelected ? 'cyan' : 'none'}
                className={`p-4 cursor-pointer transition-all ${
                  isSelected
                    ? 'ring-1 ring-cyan-400/80 bg-[#0F1B36]'
                    : 'bg-[#0A1326]/70 hover:bg-[#0E1A33]'
                }`}
              >
                <div className="flex flex-col justify-between h-full space-y-2.5">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${badgeColor}`}>
                        {scenario.threatType}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400">
                        {scenario.category.split('/')[0]}
                      </span>
                    </div>
                    <h4 className="text-sm font-bold text-white tracking-tight leading-snug">
                      {scenario.name}
                    </h4>
                    <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                      {scenario.description}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-[#1E3050]/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span>Expected: <b className="text-slate-200">{scenario.expectedCategory}</b></span>
                    <span className="text-cyan-400 flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                      {isSelected ? 'Selected' : 'Use Preset &rarr;'}
                    </span>
                  </div>
                </div>
              </CyberCard>
            );
          })}
        </div>
      </div>

      {/* 3. Scenario Inspector & Editable Payload */}
      <CyberCard className="p-5" glow="none">
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#1E3050]/70">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white font-mono">
                  Payload Editor: <span className="text-cyan-300">{selectedPreset.name}</span>
                </h3>
              </div>
              <p className="text-xs text-slate-400">{selectedPreset.whyUseful}</p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={handleCopyPayload}
                className="px-2.5 py-1.5 rounded-lg bg-[#0B152B] hover:bg-[#132247] border border-[#1E3050] text-xs font-mono text-slate-300 flex items-center gap-1.5 transition-colors"
                title="Copy raw log string"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                onClick={handleResetPayload}
                className="px-2.5 py-1.5 rounded-lg bg-[#0B152B] hover:bg-[#132247] border border-[#1E3050] text-xs font-mono text-slate-300 flex items-center gap-1.5 transition-colors"
                title="Reset to template"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
                <span>Reset Template</span>
              </button>
            </div>
          </div>

          {/* Text Area */}
          <div className="space-y-2">
            <div className="relative">
              <textarea
                value={payload}
                onChange={(e) => setPayload(e.target.value)}
                rows={4}
                placeholder="Enter raw syslog line or JSON event record..."
                className="w-full rounded-xl bg-[#070D1A] border border-[#1E3050] focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 p-3.5 text-xs font-mono text-slate-200 leading-relaxed outline-none transition-all placeholder:text-slate-600 resize-y"
              />
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 px-1">
              <span>Characters: {payload.length} | Tokens: {payload.trim().split(/\s+/).filter(Boolean).length}</span>
              <span>Input format: Auto-detected (RFC 3164 / RFC 5424 / ISO / JSON)</span>
            </div>
          </div>

          {/* Injection Action Bar */}
          <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-[#1E3050]/70">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              <span>Target: Ingestion Endpoint <code className="text-cyan-300">/api/inference/predict-logs</code></span>
            </div>

            <button
              onClick={handleInject}
              disabled={isLoading || !payload.trim()}
              className="px-6 py-2.5 rounded-xl font-bold font-mono text-xs uppercase tracking-wider text-black bg-gradient-to-r from-cyan-400 to-blue-500 hover:from-cyan-300 hover:to-blue-400 disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(6,182,212,0.4)] flex items-center justify-center gap-2 transition-all"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
                  <span>Evaluating Pipeline...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-black" />
                  <span>Inject Scenario</span>
                </>
              )}
            </button>
          </div>
        </div>
      </CyberCard>

      {/* 4. Real Pipeline Inference Results */}
      <AnimatePresence>
        {currentResult && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>Live Pipeline Detection Results</span>
              </h3>
              <span className="text-xs font-mono text-cyan-300">
                Evaluation ID: <b className="text-white">{currentResult.id}</b>
              </span>
            </div>

            {/* Top Score Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Classification Card */}
              <CyberCard className="p-4" glow={currentResult.category === 'Anomalous' ? 'rose' : currentResult.category === 'Medium' ? 'amber' : 'emerald'}>
                <div className="space-y-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    Policy Classification
                  </span>
                  <div className="flex items-center gap-2.5">
                    {currentResult.category === 'Anomalous' ? (
                      <ShieldAlert className="w-6 h-6 text-rose-400" />
                    ) : currentResult.category === 'Medium' ? (
                      <AlertTriangle className="w-6 h-6 text-amber-400" />
                    ) : (
                      <ShieldCheck className="w-6 h-6 text-emerald-400" />
                    )}
                    <span
                      className={`text-xl font-extrabold tracking-tight ${
                        currentResult.category === 'Anomalous'
                          ? 'text-rose-400'
                          : currentResult.category === 'Medium'
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }`}
                    >
                      {currentResult.category}
                    </span>
                  </div>
                  <p className="text-[11px] font-mono text-slate-400">
                    Threshold: {currentResult.category === 'Anomalous' ? '&ge; 0.70' : currentResult.category === 'Medium' ? '0.30 - 0.70' : '< 0.30'}
                  </p>
                </div>
              </CyberCard>

              {/* Anomaly Probability Card */}
              <CyberCard className="p-4" glow="none">
                <div className="space-y-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    Anomaly Probability
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-extrabold text-white font-mono">
                      {currentResult.probability !== null && currentResult.probability !== undefined
                        ? `${(currentResult.probability * 100).toFixed(2)}%`
                        : 'N/A'}
                    </span>
                    <span className="text-xs font-mono text-slate-500">
                      ({currentResult.probability?.toFixed(4) ?? 'N/A'})
                    </span>
                  </div>
                  <div className="w-full bg-[#0A1326] h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        (currentResult.probability ?? 0) >= 0.7
                          ? 'bg-rose-500'
                          : (currentResult.probability ?? 0) >= 0.3
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, (currentResult.probability ?? 0) * 100))}%` }}
                    />
                  </div>
                </div>
              </CyberCard>

              {/* Model Confidence Card */}
              <CyberCard className="p-4" glow="none">
                <div className="space-y-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    Decision Confidence
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-extrabold text-cyan-300 font-mono">
                      {currentResult.confidence !== null && currentResult.confidence !== undefined
                        ? `${(currentResult.confidence * 100).toFixed(1)}%`
                        : 'N/A'}
                    </span>
                    <span className="text-xs font-mono text-slate-500">Distance from 0.5</span>
                  </div>
                  <p className="text-[11px] font-mono text-slate-400">
                    {currentResult.category === 'Medium' ? 'Ambiguous boundary region' : 'Decisive model split'}
                  </p>
                </div>
              </CyberCard>

              {/* Originating Service Card */}
              <CyberCard className="p-4" glow="none">
                <div className="space-y-2">
                  <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    Parsed Service / Daemon
                  </span>
                  <div className="flex items-center gap-2">
                    <Server className="w-5 h-5 text-cyan-400 shrink-0" />
                    <span className="text-base font-bold text-white font-mono truncate">
                      {currentResult.service || 'unknown'}
                    </span>
                  </div>
                  <p className="text-[11px] font-mono text-slate-400 truncate">
                    Host: {currentResult.hostname || 'localhost'}
                  </p>
                </div>
              </CyberCard>
            </div>

            {/* Extracted Feature Summary Pills */}
            <CyberCard className="p-4" glow="none">
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
                    <Hash className="w-3.5 h-3.5 text-cyan-400" />
                    Extracted Canonical Features (Inference Vector)
                  </span>
                  <span className="text-slate-500">11–24 Feature Vector (Behavioral Mode)</span>
                </div>

                <div className="flex flex-wrap gap-2 text-xs font-mono">
                  <div className="px-2.5 py-1 rounded bg-[#091124] border border-[#1E3050] text-slate-300">
                    Service: <b className="text-cyan-300">{currentResult.service || 'unknown'}</b>
                  </div>
                  <div className={`px-2.5 py-1 rounded border ${currentResult.observed_facts?.has_error_flag ? 'bg-rose-950/60 border-rose-500/50 text-rose-300' : 'bg-[#091124] border-[#1E3050] text-slate-400'}`}>
                    has_error: <b>{currentResult.observed_facts?.has_error_flag ? '1 (True)' : '0 (False)'}</b>
                  </div>
                  <div className={`px-2.5 py-1 rounded border ${currentResult.observed_facts?.has_auth_flag ? 'bg-amber-950/60 border-amber-500/50 text-amber-300' : 'bg-[#091124] border-[#1E3050] text-slate-400'}`}>
                    has_auth: <b>{currentResult.observed_facts?.has_auth_flag ? '1 (True)' : '0 (False)'}</b>
                  </div>
                  <div className={`px-2.5 py-1 rounded border ${currentResult.observed_facts?.has_failed_auth_flag ? 'bg-rose-950/60 border-rose-500/50 text-rose-300' : 'bg-[#091124] border-[#1E3050] text-slate-400'}`}>
                    has_failed_auth: <b>{currentResult.observed_facts?.has_failed_auth_flag ? '1 (True)' : '0 (False)'}</b>
                  </div>
                  <div className={`px-2.5 py-1 rounded border ${currentResult.observed_facts?.has_denied_flag ? 'bg-rose-950/60 border-rose-500/50 text-rose-300' : 'bg-[#091124] border-[#1E3050] text-slate-400'}`}>
                    has_denied: <b>{currentResult.observed_facts?.has_denied_flag ? '1 (True)' : '0 (False)'}</b>
                  </div>
                  <div className="px-2.5 py-1 rounded bg-[#091124] border border-[#1E3050] text-slate-300">
                    has_ip: <b>{currentResult.observed_facts?.detected_source_ip ? `1 (${currentResult.observed_facts.detected_source_ip})` : '0 (None)'}</b>
                  </div>
                  <div className="px-2.5 py-1 rounded bg-[#091124] border border-[#1E3050] text-slate-300">
                    Entropy: <b className="text-cyan-300">{currentResult.observed_facts?.entropy?.toFixed(4) ?? 'N/A'}</b>
                  </div>
                  <div className="px-2.5 py-1 rounded bg-[#091124] border border-[#1E3050] text-slate-300">
                    Tokens: <b className="text-cyan-300">{currentResult.observed_facts?.token_count ?? 'N/A'}</b>
                  </div>
                  <div className="px-2.5 py-1 rounded bg-[#091124] border border-[#1E3050] text-slate-300">
                    Length: <b className="text-cyan-300">{currentResult.observed_facts?.message_length ?? currentResult.raw_log.length} chars</b>
                  </div>
                </div>
              </div>
            </CyberCard>

            {/* Tabbed Explanation & SHAP Section */}
            <CyberCard className="p-5" glow="none">
              <div className="space-y-4">
                {/* Tab Navigation */}
                <div className="flex items-center gap-2 border-b border-[#1E3050]/70 pb-3">
                  <button
                    onClick={() => setActiveTab('explanation')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                      activeTab === 'explanation'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Plain-English Decision Rationale
                  </button>
                  <button
                    onClick={() => setActiveTab('shap')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                      activeTab === 'shap'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    SHAP Mathematical Attributions
                  </button>
                  <button
                    onClick={() => setActiveTab('facts')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                      activeTab === 'facts'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Observed Facts & Security Context
                  </button>
                  <button
                    onClick={() => setActiveTab('raw')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                      activeTab === 'raw'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Raw Payload
                  </button>
                </div>

                {/* Tab 1: Plain English Explanation */}
                {activeTab === 'explanation' && (
                  <div className="space-y-4">
                    <div className="p-4 rounded-xl bg-[#070D1A] border border-[#1E3050] space-y-2">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400">
                        Primary Decision Rationale
                      </span>
                      <p className="text-sm font-sans text-slate-100 leading-relaxed">
                        {currentResult.plain_english?.why ||
                          currentResult.model_explanation?.decision_rationale ||
                          currentResult.explanation_summary ||
                          'The model evaluated the event against its learned baseline operational parameters.'}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="p-3.5 rounded-xl bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                          How Factors Influenced Model
                        </span>
                        <p className="text-xs text-slate-300 leading-relaxed">
                          {currentResult.plain_english?.how_factors_influenced ||
                            'Feature values were transformed and evaluated against the decision trees in the ensemble.'}
                        </p>
                      </div>

                      <div className="p-3.5 rounded-xl bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                          Recommended Operational Action
                        </span>
                        <p className="text-xs text-slate-300 leading-relaxed">
                          {currentResult.plain_english?.recommended_action ||
                            currentResult.security_interpretation ||
                            'Monitor subsequent activity from this host and originating service.'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 2: SHAP Attributions Bar Chart */}
                {activeTab === 'shap' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                      <span>Feature Impact Breakdown (Game-Theoretic Shapley Values)</span>
                      <span className="text-slate-500">
                        Base Value: <code className="text-cyan-300">{currentResult.shap?.base_value?.toFixed(4) ?? '0.4120'}</code>
                      </span>
                    </div>

                    {currentResult.shap && currentResult.shap.available && currentResult.shap.contributions && currentResult.shap.contributions.length > 0 ? (
                      <div className="space-y-2">
                        {currentResult.shap.contributions.slice(0, 8).map((c: ShapContribution, idx: number) => {
                          const isPushingAnomaly = c.direction === 'anomaly' || c.contribution > 0;
                          const absVal = Math.abs(c.contribution);
                          const maxContribution = Math.max(...currentResult.shap!.contributions.map((x) => Math.abs(x.contribution)), 0.1);
                          const barWidthPercent = Math.min(100, Math.max(8, (absVal / maxContribution) * 100));

                          return (
                            <div key={idx} className="p-2.5 rounded-lg bg-[#070D1A] border border-[#1E3050] space-y-1.5">
                              <div className="flex items-center justify-between text-xs font-mono">
                                <div className="flex items-center gap-2">
                                  <span className={`w-2 h-2 rounded-full ${isPushingAnomaly ? 'bg-rose-400' : 'bg-emerald-400'}`} />
                                  <span className="text-white font-bold">{c.feature}</span>
                                  <span className="text-slate-500">({String(c.feature_value ?? '')})</span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                                      isPushingAnomaly
                                        ? 'bg-rose-950/70 text-rose-300 border border-rose-500/40'
                                        : 'bg-emerald-950/70 text-emerald-300 border border-emerald-500/40'
                                    }`}
                                  >
                                    {isPushingAnomaly ? '+ Pushes Anomaly' : '- Pushes Normal'}
                                  </span>
                                  <span className="font-bold text-slate-200">
                                    {c.contribution > 0 ? `+${c.contribution.toFixed(4)}` : c.contribution.toFixed(4)}
                                  </span>
                                </div>
                              </div>

                              {/* Visual Attribution Bar */}
                              <div className="w-full bg-[#0D1526] h-2 rounded-full overflow-hidden flex">
                                <div
                                  className={`h-full rounded-full ${isPushingAnomaly ? 'bg-gradient-to-r from-rose-500 to-rose-400' : 'bg-gradient-to-r from-emerald-500 to-emerald-400'}`}
                                  style={{ width: `${barWidthPercent}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="p-6 rounded-xl bg-[#070D1A] border border-[#1E3050] text-center space-y-2">
                        <HelpCircle className="w-6 h-6 text-slate-500 mx-auto" />
                        <p className="text-xs text-slate-400 font-mono">
                          {currentResult.shap?.error || 'Feature-level SHAP explanation was not computed for this record.'}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Tab 3: Observed Facts */}
                {activeTab === 'facts' && (
                  <div className="space-y-3 text-xs font-mono">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-slate-500">Raw Severity Code</span>
                        <p className="text-white font-bold">{currentResult.observed_facts?.raw_severity_code ?? currentResult.severity ?? '6 (Information)'}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-slate-500">Extracted User</span>
                        <p className="text-white font-bold">{currentResult.observed_facts?.detected_user || 'None'}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-slate-500">Source IP Address</span>
                        <p className="text-white font-bold">{currentResult.observed_facts?.detected_source_ip || 'None detected'}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-[#070D1A] border border-[#1E3050] space-y-1">
                        <span className="text-slate-500">Suspicious Context Flag</span>
                        <p className="text-white font-bold">{currentResult.observed_facts?.has_suspicious_context ? 'Yes (Evasion Signature)' : 'No'}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Tab 4: Raw Payload */}
                {activeTab === 'raw' && (
                  <div className="p-3.5 rounded-xl bg-[#070D1A] border border-[#1E3050]">
                    <pre className="text-xs font-mono text-cyan-200 whitespace-pre-wrap break-all leading-relaxed">
                      {currentResult.raw_log}
                    </pre>
                  </div>
                )}
              </div>
            </CyberCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 5. Simulation Session History */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span>Session Simulation History</span>
          </h3>
          <span className="text-xs text-slate-500 font-mono">
            {history.length} Injections Recorded
          </span>
        </div>

        {history.length === 0 ? (
          <EmptyState
            icon={Zap}
            title="No Injections Yet"
            description="Select one of the 6 pre-calibrated scenario presets above and click 'Inject Scenario' to test the real inference pipeline."
          />
        ) : (
          <div className="space-y-2">
            {history.map((item) => (
              <CyberCard
                key={item.id}
                className="p-3 hover:bg-[#0E1A33] transition-colors"
                glow="none"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-bold truncate">{item.scenarioName}</span>
                      <span className="text-slate-500">[{item.category.split('/')[0]}]</span>
                    </div>
                    <p className="text-slate-400 text-[11px] truncate">{item.payloadSnippet}</p>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <StatusBadge status={item.classification} size="sm" />
                    <span className="text-slate-300 font-bold">
                      {item.probability !== null ? `${(item.probability * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                    <span className="text-slate-500 text-[11px]">{item.timestamp}</span>
                    <button
                      onClick={() => handleInspectHistory(item)}
                      className="px-2.5 py-1 rounded bg-[#091224] hover:bg-cyan-950/60 border border-[#1E3050] hover:border-cyan-500/50 text-cyan-300 text-[11px] font-mono transition-colors"
                    >
                      Inspect
                    </button>
                  </div>
                </div>
              </CyberCard>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
export default SimulationLabPage;
