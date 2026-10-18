import os
import sys
import time
import socket
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import psutil

logger = logging.getLogger("smartlog.collector.process")

# Standard tracked interactive and administrative applications
TARGET_APPLICATIONS = {
    "notepad.exe": "Notepad Text Editor",
    "notepad": "Notepad Text Editor",
    "cmd.exe": "Windows Command Prompt",
    "cmd": "Windows Command Prompt",
    "powershell.exe": "Windows PowerShell",
    "powershell": "Windows PowerShell",
    "pwsh.exe": "PowerShell Core",
    "pwsh": "PowerShell Core",
    "control.exe": "Windows Control Panel",
    "control": "Windows Control Panel",
    "systemsettings.exe": "Windows Settings / Control Panel",
    "systemsettings": "Windows Settings / Control Panel",
    "taskmgr.exe": "Windows Task Manager",
    "taskmgr": "Windows Task Manager",
    "calc.exe": "Windows Calculator",
    "calc": "Windows Calculator",
    "calculatorapp.exe": "Windows Calculator",
    "regedit.exe": "Windows Registry Editor",
    "regedit": "Windows Registry Editor",
    "mmc.exe": "Microsoft Management Console",
    "services.msc": "Services Manager",
    "wt.exe": "Windows Terminal",
    "explorer.exe": "Windows Explorer",
    "explorer": "Windows Explorer"
}

# Suspicious argument signatures (non-malicious launches remain Normal)
SUSPICIOUS_ARG_KEYWORDS = [
    "-enc", "-encodedcommand", "downloadstring", "invoke-expression",
    "iex", "bypass", "-nop", "-noprofile", "-w hidden", "-windowstyle hidden",
    "mimikatz", "sekurlsa", "lsadump", "procdump", "vssadmin delete shadows"
]

class ProcessLifecycleMonitor:
    """Monitors Windows process creation and termination in near real-time.
    
    Tracks application open/close events for target applications (Notepad, CMD,
    Control Panel, PowerShell, Explorer, etc.) and system processes, normalizing
    them into standard SOC log events.
    """

    def __init__(self, hostname: Optional[str] = None):
        self.hostname = hostname or os.environ.get("COMPUTERNAME") or socket.gethostname()
        self._lock = threading.RLock()
        self._tracked_processes: Dict[int, Dict[str, Any]] = {}
        self._is_initialized = False
        self._synthetic_record_counter = int(time.time() * 1000) % 10000000

    def _next_record_id(self) -> int:
        with self._lock:
            self._synthetic_record_counter += 1
            return self._synthetic_record_counter

    def initialize_baseline(self) -> List[Dict[str, Any]]:
        """Take an initial snapshot of running processes and detect already-active target applications."""
        initial_events = []
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        with self._lock:
            snapshot = {}
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    info = proc.info
                    pid = info.get('pid')
                    if not pid or pid <= 4:
                        continue
                    p_name = info.get('name') or "unknown.exe"
                    p_lower = p_name.lower()
                    p_base = p_lower.replace(".exe", "")
                    is_target = (p_lower in TARGET_APPLICATIONS) or (p_base in TARGET_APPLICATIONS)

                    if is_target:
                        detail = self._get_detailed_process_info(pid)
                        snapshot[pid] = detail
                        app_title = TARGET_APPLICATIONS.get(p_lower, TARGET_APPLICATIONS.get(p_base, p_name))
                        rec_id = self._next_record_id()
                        msg = (
                            f"[APPLICATION_OPENED] {app_title} ('{p_name}') active on system (PID: {pid}, user: '{detail['username']}'). "
                            f"Executable: {detail['exe'] or 'System Path'}"
                        )
                        raw_log = f"{now_iso} {self.hostname} {p_name}[{pid}]: {msg}"
                        initial_events.append({
                            "channel": "ProcessMonitor",
                            "record_id": rec_id,
                            "event_id": 4688,
                            "provider": p_name,
                            "computer": self.hostname,
                            "timestamp": now_iso,
                            "severity": 6,
                            "severity_label": "Information",
                            "message": msg,
                            "user_id": detail["username"],
                            "pid": pid,
                            "ppid": detail.get("ppid"),
                            "process_name": p_name,
                            "executable_path": detail["exe"],
                            "cmdline": detail["cmdline"],
                            "action": "APPLICATION_OPENED",
                            "event_type": "APPLICATION_OPENED",
                            "is_target_app": True,
                            "has_suspicious_context": False,
                            "raw": raw_log
                        })
                    else:
                        snapshot[pid] = {
                            "name": p_name,
                            "create_time": time.time(),
                            "exe": "",
                            "username": "SYSTEM",
                            "cmdline": ""
                        }
                except Exception:
                    continue
            self._tracked_processes = snapshot
            self._is_initialized = True
            logger.info("ProcessLifecycleMonitor baseline initialized with %d processes (%d active target apps).", len(snapshot), len(initial_events))
        return initial_events

    def _get_detailed_process_info(self, pid: int) -> Dict[str, Any]:
        """Fetch metadata safely for newly appeared processes."""
        if pid <= 4:
            return {
                "pid": pid, "ppid": None, "name": "System" if pid == 4 else "Idle",
                "exe": "", "cmdline": "", "username": "SYSTEM", "create_time": time.time()
            }

        name = "unknown.exe"
        exe = ""
        cmdline = ""
        username = "SYSTEM"
        ppid = None
        create_time = time.time()
        try:
            p = psutil.Process(pid)
            try:
                name = p.name() or "unknown.exe"
            except Exception:
                pass

            is_target = name.lower() in TARGET_APPLICATIONS
            if is_target:
                try:
                    exe = p.exe() or ""
                except Exception:
                    pass
                try:
                    username = p.username() or "SYSTEM"
                except Exception:
                    pass
                try:
                    ppid = p.ppid()
                except Exception:
                    pass
                if any(x in name.lower() for x in ("cmd", "powershell", "pwsh")):
                    try:
                        cmd_list = p.cmdline()
                        cmdline = " ".join(cmd_list) if cmd_list else ""
                    except Exception:
                        pass
            else:
                try:
                    ppid = p.ppid()
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "pid": pid,
            "ppid": ppid,
            "name": name,
            "exe": exe,
            "cmdline": cmdline,
            "username": username,
            "create_time": create_time
        }

    def poll(self) -> List[Dict[str, Any]]:
        """Fast PID set comparison against previous snapshot.
        
        Emits APPLICATION_OPENED / PROCESS_CREATED for new processes, and
        APPLICATION_CLOSED / PROCESS_TERMINATED for exited processes.
        """
        if not self._is_initialized:
            self.initialize_baseline()
            return []

        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        events: List[Dict[str, Any]] = []

        try:
            current_pids = {p for p in psutil.pids() if p > 4}
        except Exception as e:
            logger.error("Failed to fetch running PIDs: %s", e)
            return []

        with self._lock:
            previous_pids = set(self._tracked_processes.keys())

            # Detect NEW processes (APPLICATION_OPENED)
            new_pids = current_pids - previous_pids
            for pid in new_pids:
                detail = self._get_detailed_process_info(pid)
                proc_name = detail["name"]
                proc_name_lower = proc_name.lower()
                exe_path = detail["exe"] or "Unknown Path"
                username = detail["username"] or "Unknown User"
                cmdline = detail["cmdline"]

                # Determine whether process is a target application
                p_base = proc_name_lower.replace(".exe", "")
                is_target_app = (proc_name_lower in TARGET_APPLICATIONS) or (p_base in TARGET_APPLICATIONS)
                app_title = TARGET_APPLICATIONS.get(proc_name_lower, TARGET_APPLICATIONS.get(p_base, proc_name))

                # Check for suspicious command-line context
                cmd_lower = cmdline.lower()
                has_suspicious_args = any(sig in cmd_lower for sig in SUSPICIOUS_ARG_KEYWORDS)
                is_temp_execution = "appdata\\local\\temp" in exe_path.lower() or "windows\\temp" in exe_path.lower()
                
                action_name = "APPLICATION_OPENED" if is_target_app else "PROCESS_CREATED"
                event_id = 4688  # Windows Security Audit Process Creation standard ID

                rec_id = self._next_record_id()
                
                if is_target_app:
                    msg = (
                        f"[{action_name}] {app_title} ('{proc_name}') opened by user '{username}' (PID: {pid}). "
                        f"Executable: {exe_path}"
                    )
                else:
                    msg = (
                        f"[{action_name}] Process '{proc_name}' (PID: {pid}) started by user '{username}'. "
                        f"Executable: {exe_path}"
                    )

                if cmdline:
                    msg += f" | Command Line: {cmdline[:200]}"

                if has_suspicious_args or is_temp_execution:
                    msg += " [Context Warning: Suspicious execution parameters or path detected]"

                raw_log = f"{now_iso} {self.hostname} {proc_name}[{pid}]: {msg}"

                events.append({
                    "channel": "ProcessMonitor",
                    "record_id": rec_id,
                    "event_id": event_id,
                    "provider": proc_name,
                    "computer": self.hostname,
                    "timestamp": now_iso,
                    "severity": 4 if (has_suspicious_args or is_temp_execution) else 6,
                    "severity_label": "Warning" if (has_suspicious_args or is_temp_execution) else "Information",
                    "message": msg,
                    "user_id": username,
                    "pid": pid,
                    "ppid": detail.get("ppid"),
                    "process_name": proc_name,
                    "executable_path": exe_path,
                    "cmdline": cmdline,
                    "action": action_name,
                    "event_type": action_name,
                    "is_target_app": is_target_app,
                    "has_suspicious_context": (has_suspicious_args or is_temp_execution),
                    "raw": raw_log
                })

                # Register in tracked processes
                self._tracked_processes[pid] = detail

            # Detect TERMINATED processes (APPLICATION_CLOSED)
            closed_pids = previous_pids - current_pids
            for pid in closed_pids:
                old_info = self._tracked_processes.pop(pid, {})
                proc_name = old_info.get("name") or "unknown.exe"
                proc_name_lower = proc_name.lower()
                username = old_info.get("username") or "Unknown User"
                exe_path = old_info.get("exe") or "Unknown Path"
                create_time = old_info.get("create_time", time.time())
                duration_seconds = round(max(0.0, time.time() - create_time), 1)

                p_base = proc_name_lower.replace(".exe", "")
                is_target_app = (proc_name_lower in TARGET_APPLICATIONS) or (p_base in TARGET_APPLICATIONS)
                app_title = TARGET_APPLICATIONS.get(proc_name_lower, TARGET_APPLICATIONS.get(p_base, proc_name))
                action_name = "APPLICATION_CLOSED" if is_target_app else "PROCESS_TERMINATED"
                event_id = 4689  # Windows Security Audit Process Exit standard ID

                rec_id = self._next_record_id()
                
                if is_target_app:
                    msg = (
                        f"[{action_name}] {app_title} ('{proc_name}') closed (PID: {pid}, user: '{username}'). "
                        f"Total runtime duration: {duration_seconds}s"
                    )
                else:
                    msg = (
                        f"[{action_name}] Process '{proc_name}' (PID: {pid}) terminated. "
                        f"Total runtime duration: {duration_seconds}s"
                    )

                raw_log = f"{now_iso} {self.hostname} {proc_name}[{pid}]: {msg}"

                events.append({
                    "channel": "ProcessMonitor",
                    "record_id": rec_id,
                    "event_id": event_id,
                    "provider": proc_name,
                    "computer": self.hostname,
                    "timestamp": now_iso,
                    "severity": 6,
                    "severity_label": "Information",
                    "message": msg,
                    "user_id": username,
                    "pid": pid,
                    "ppid": old_info.get("ppid"),
                    "process_name": proc_name,
                    "executable_path": exe_path,
                    "action": action_name,
                    "event_type": action_name,
                    "duration_seconds": duration_seconds,
                    "is_target_app": is_target_app,
                    "has_suspicious_context": False,
                    "raw": raw_log
                })

        if events:
            logger.info("ProcessLifecycleMonitor detected %d process lifecycle events.", len(events))

        return events
