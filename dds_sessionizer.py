"""
DDS Hybrid Session Detector
============================
Replaces the fragile MCE-ON-only sessionization in dds_cross_session.py.

Key findings from data analysis across 4 locos:

  IRPRP30821: MCE ON coverage only 36% (43 ON vs 120 OFF)
              235 large gaps with NO power-down signal before them
              — loco was running cleanly, DDS just had no events

  IRPRP42012: MCE ON coverage 66% (51 ON vs 77 OFF)

  IRPRP43771: MCE ON coverage 102% — well-behaved log

  IRPRP37571: MCE ON coverage 96% — well-behaved log

Root cause of missing MCE ON events:
  - DDS buffer being downloaded at the exact moment of power-on
  - Loco powered on before the logging window began
  - Some export configurations filter it out
  - 'S:0110-Process enable signal is FALSE' appears as a power-on proxy

Core design principle:
  A session boundary is ONLY valid if we have positive evidence of a
  power cycle. A bare silence gap is NOT a session boundary — it just
  means the loco was running without generating notable events.

  The only two things that definitively bound a session are:
    1. An explicit MCE ON event  (primary)
    2. A power-down signal followed by a gap, then any event  (secondary)
       where power-down = MCE OFF, Process enable FALSE, SS01 isolation,
       or fast/protective shutdown cluster

  Gap alone → NOT a boundary. We merge the gap into the running session.

Session types assigned during detection:
  OPERATIONAL   — loco under power, traction/brake events present
  FAULT_ACTIVE  — P1/P2 faults dominating, loco struggling
  INVESTIGATION — repeated same-fault-on-restart pattern
  TEST          — deliberate isolation/shunting/ZTEL sequences
  IDLE          — powered but no significant events (parked, energy saving)
  INCOMPLETE    — boundary conditions at start/end of file
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Signals that definitively mark MCE power-on when the explicit event is absent
MCE_ON_PROXIES = [
    "Power on of MCE",            # primary — explicit
    "S:0110-Process enable",      # CCUO starts processing — loco coming online
    "CCUO:0087",                  # ECode for MCE ON in some export variants
]

# Signals that definitively mark MCE power-down
MCE_OFF_SIGNALS = [
    "Power Off MCE",
    "MCE off",
    "SS01 main power off",
    "Process enable signal is FALSE",
    "S:0110-Process enable signal is FALSE",
]

# Secondary power-down evidence — less definitive, used in combination
POWERDOWN_CLUSTER = [
    "Protective shutdown",
    "Fast shutdown",
    "SS01 main power off",
    "SS02 traction bogie1 off",
    "SS03 traction bogie2 off",
    "FPGA caused PS",
    "Subsystem 02 & 03 off",
]

# Operational context signals — presence means loco is under power and moving/working
OPERATIONAL_SIGNALS = [
    "Shunting mode",
    "banking operation",
    "brake cock",
    "loco brake",
    "auto brake",
    "emgbrk",
    "main res",
    "AFL Activated",
    "ACP/Train",
    "vigilance",
    "Overspeed",
    "Electrical slip",
    "Torque reduct",
    "Power limitation",
    "Pantograph bouncing",
    "Catenary Voltage",
    "Primary voltage",
    "Line volt",
    "OHE",
]

# Test/deliberate isolation signals
TEST_SIGNALS = [
    "Shunting mode",
    "ZTEL operated",
    "Simulation",
    "banking operation",
    "Rotary switch bogie",
    "Iso Request CON",
]

# Minimum gap in minutes to even consider a session break
# Below this: just normal event spacing within a running session
MIN_SESSION_GAP_MIN = 120

# A power-down cluster is valid if N or more POWERDOWN_CLUSTER events
# appear within this many minutes
POWERDOWN_CLUSTER_WINDOW_MIN = 5
POWERDOWN_CLUSTER_MIN_COUNT  = 2


# ---------------------------------------------------------------------------
# DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_n:     int
    start_time:    pd.Timestamp
    end_time:      pd.Timestamp
    df:            pd.DataFrame         # events within this session
    session_type:  str = "UNKNOWN"
    boundary_type: str = "UNKNOWN"      # how start was detected
    duration_min:  float = 0.0
    n_events:      int = 0
    p1_count:      int = 0
    p2_count:      int = 0
    gap_before_min: Optional[float] = None   # gap from previous session end
    is_fault_driven_end: bool = False
    fault_driven_evidence: str = ""


# ---------------------------------------------------------------------------
# POWER-DOWN DETECTOR
# ---------------------------------------------------------------------------

def _detect_powerdown(window_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Return (is_powerdown, evidence) for a window of events.
    A powerdown is confirmed if:
      a) MCE OFF signal present, OR
      b) Two or more POWERDOWN_CLUSTER events within POWERDOWN_CLUSTER_WINDOW_MIN
    """
    texts = window_df["Dist Text"].fillna("").tolist()

    # Explicit MCE off
    for sig in MCE_OFF_SIGNALS:
        if any(sig.lower() in t.lower() for t in texts):
            return True, f"Explicit power-down: '{sig}'"

    # Cluster of protective shutdown events
    cluster_hits = []
    for i, row in window_df.iterrows():
        t = str(row.get("Dist Text", ""))
        for sig in POWERDOWN_CLUSTER:
            if sig.lower() in t.lower():
                cluster_hits.append(row["Start time"])
                break

    if len(cluster_hits) >= POWERDOWN_CLUSTER_MIN_COUNT:
        span = (cluster_hits[-1] - cluster_hits[0]).total_seconds() / 60
        if span <= POWERDOWN_CLUSTER_WINDOW_MIN:
            return True, f"Shutdown cluster: {len(cluster_hits)} events in {span:.1f}min"

    return False, ""


# ---------------------------------------------------------------------------
# SESSION TYPE CLASSIFIER
# ---------------------------------------------------------------------------

def _classify_session(sdf: pd.DataFrame, duration_min: float) -> str:
    """Classify a session by its dominant event character."""
    if sdf.empty:
        return "IDLE"

    texts  = sdf["Dist Text"].fillna("").str.lower()
    n_p1   = (sdf["Prio"] == 1).sum()
    n_p2   = (sdf["Prio"] == 2).sum()
    n_tot  = len(sdf)

    # Test: deliberate isolation language
    for sig in TEST_SIGNALS:
        if texts.str.contains(sig.lower(), regex=False).any():
            return "TEST"

    # Investigation: same P1 fault repeating in short session
    if duration_min < 60 and n_p1 >= 2:
        top = sdf[sdf["Prio"] == 1]["Dist Text"].value_counts()
        if len(top) and top.iloc[0] >= 2:
            return "INVESTIGATION"

    # Fault active: P1/P2 heavy
    if n_p1 + n_p2 > n_tot * 0.4:
        return "FAULT_ACTIVE"

    # Operational: traction/brake context
    for sig in OPERATIONAL_SIGNALS:
        if texts.str.contains(sig.lower(), regex=False).any():
            return "OPERATIONAL"

    # Parked idle: long, sparse, no real faults
    if duration_min > 300 and n_tot < 20:
        return "IDLE"

    return "OPERATIONAL"


# ---------------------------------------------------------------------------
# HYBRID SESSION DETECTOR
# ---------------------------------------------------------------------------

def detect_sessions(df: pd.DataFrame) -> List[Session]:
    """
    Main entry point. Takes a sorted DDS DataFrame and returns a list of
    Session objects with accurate boundaries, types, and power-cycle evidence.

    Algorithm:
      1. Collect all explicit MCE ON timestamps as primary boundaries.
      2. For each gap > MIN_SESSION_GAP_MIN that does NOT contain an MCE ON,
         check whether there is a power-down signal in the tail of the
         preceding block. If yes → treat as a secondary session boundary.
         If no → merge the gap into the surrounding session.
      3. Classify each resulting session.
      4. Mark fault-driven vs clean ends.
    """
    if df.empty:
        return []

    df = df.sort_values("Start time").reset_index(drop=True)

    # ---- Step 1: Collect all candidate boundary timestamps ----

    # Primary: explicit MCE ON
    primary_starts = set()
    for _, row in df.iterrows():
        t = str(row.get("Dist Text", ""))
        if any(proxy.lower() in t.lower() for proxy in MCE_ON_PROXIES):
            primary_starts.add(row["Start time"])

    # Secondary: large gaps where the tail of preceding block has a power-down
    gaps_min = df["Start time"].diff().dt.total_seconds() / 60
    candidate_boundaries: List[Tuple[pd.Timestamp, str]] = []

    for idx in df.index[gaps_min > MIN_SESSION_GAP_MIN]:
        t_new    = df.loc[idx, "Start time"]
        gap_val  = gaps_min[idx]

        # Skip if this is already covered by a primary boundary nearby (±10min)
        if any(abs((t_new - p).total_seconds()) < 600 for p in primary_starts):
            continue

        # Look at the 10 events before this gap for power-down evidence
        tail = df.loc[max(0, idx - 10) : idx - 1]
        is_down, evidence = _detect_powerdown(tail)

        if is_down:
            candidate_boundaries.append((t_new, f"Gap {gap_val:.0f}min + {evidence}"))

    # Merge all boundaries
    all_boundaries: List[Tuple[pd.Timestamp, str]] = (
        [(t, "MCE ON event") for t in sorted(primary_starts)] +
        candidate_boundaries
    )
    all_boundaries.sort(key=lambda x: x[0])

    # If no boundaries found at all, the whole file is one session
    if not all_boundaries:
        sdf = df.copy()
        dur = (df["Start time"].max() - df["Start time"].min()).total_seconds() / 60
        s = Session(
            session_n      = 1,
            start_time     = df["Start time"].min(),
            end_time       = df["Start time"].max(),
            df             = sdf,
            boundary_type  = "WHOLE_FILE",
            duration_min   = dur,
            n_events       = len(sdf),
            p1_count       = int((sdf["Prio"] == 1).sum()),
            p2_count       = int((sdf["Prio"] == 2).sum()),
        )
        s.session_type = _classify_session(sdf, dur)
        return [s]

    # ---- Step 2: Slice DataFrame into sessions ----

    sessions: List[Session] = []
    boundary_times = [b[0] for b in all_boundaries]
    boundary_types = {b[0]: b[1] for b in all_boundaries}

    # Handle events BEFORE the first boundary as session 0 (INCOMPLETE)
    file_start = df["Start time"].min()
    if boundary_times[0] > file_start:
        pre_df  = df[df["Start time"] < boundary_times[0]].copy()
        pre_dur = (boundary_times[0] - file_start).total_seconds() / 60
        if len(pre_df) > 0:
            s = Session(
                session_n      = 0,
                start_time     = file_start,
                end_time       = boundary_times[0],
                df             = pre_df,
                boundary_type  = "INCOMPLETE_START",
                duration_min   = pre_dur,
                n_events       = len(pre_df),
                p1_count       = int((pre_df["Prio"] == 1).sum()),
                p2_count       = int((pre_df["Prio"] == 2).sum()),
            )
            s.session_type = _classify_session(pre_df, pre_dur)
            sessions.append(s)

    # Main sessions
    for i, t_start in enumerate(boundary_times):
        t_end = (boundary_times[i + 1]
                 if i + 1 < len(boundary_times)
                 else df["Start time"].max() + pd.Timedelta(seconds=1))

        sdf = df[(df["Start time"] >= t_start) & (df["Start time"] < t_end)].copy()
        if sdf.empty:
            continue

        dur = (t_end - t_start).total_seconds() / 60

        # Gap before this session (from end of previous session)
        gap_before = None
        if sessions:
            prev_end  = sessions[-1].end_time
            gap_before = (t_start - prev_end).total_seconds() / 60

        # Fault-driven end?
        tail_5min = sdf[sdf["Start time"] >= sdf["Start time"].max() - pd.Timedelta(minutes=5)]
        fd_end, fd_ev = _detect_powerdown(tail_5min)
        if not fd_end:
            p1p2_tail = tail_5min[tail_5min["Prio"].isin([1, 2])]
            if not p1p2_tail.empty:
                fd_end = True
                fd_ev  = f"P{int(p1p2_tail.iloc[-1]['Prio'])} fault before end"

        s = Session(
            session_n             = i + 1,
            start_time            = t_start,
            end_time              = t_end,
            df                    = sdf,
            boundary_type         = boundary_types.get(t_start, "UNKNOWN"),
            duration_min          = dur,
            n_events              = len(sdf),
            p1_count              = int((sdf["Prio"] == 1).sum()),
            p2_count              = int((sdf["Prio"] == 2).sum()),
            gap_before_min        = gap_before,
            is_fault_driven_end   = fd_end,
            fault_driven_evidence = fd_ev,
        )
        s.session_type = _classify_session(sdf, dur)
        sessions.append(s)

    return sessions


# ---------------------------------------------------------------------------
# COVERAGE REPORT
# ---------------------------------------------------------------------------

def session_coverage_report(sessions: List[Session], df: pd.DataFrame) -> str:
    """
    Print a summary of session detection quality for verification.
    """
    lines = []
    total_events   = len(df)
    covered_events = sum(s.n_events for s in sessions)
    pct_covered    = covered_events / total_events * 100 if total_events else 0

    primary_count   = sum(1 for s in sessions if "MCE ON" in s.boundary_type)
    secondary_count = sum(1 for s in sessions if "Gap" in s.boundary_type)
    incomplete      = sum(1 for s in sessions if "INCOMPLETE" in s.boundary_type)
    fault_ended     = sum(1 for s in sessions if s.is_fault_driven_end)

    lines.append(f"  Sessions detected  : {len(sessions)}")
    lines.append(f"  Primary (MCE ON)   : {primary_count}")
    lines.append(f"  Secondary (gap+PD) : {secondary_count}")
    lines.append(f"  Incomplete         : {incomplete}")
    lines.append(f"  Fault-driven ends  : {fault_ended}")
    lines.append(f"  Event coverage     : {covered_events}/{total_events} ({pct_covered:.0f}%)")
    lines.append("")

    type_counts = {}
    for s in sessions:
        type_counts[s.session_type] = type_counts.get(s.session_type, 0) + 1
    lines.append("  Session types:")
    for stype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {stype:<18} {cnt}")

    lines.append("")
    lines.append("  Session list (first 20):")
    for s in sessions[:20]:
        fault_marker = " [FAULT-END]" if s.is_fault_driven_end else ""
        lines.append(
            f"    S{s.session_n:3d} | {s.start_time.strftime('%d%b %H:%M')} | "
            f"{s.duration_min:6.0f}min | {s.session_type:<16} | "
            f"ev={s.n_events:3d} P1={s.p1_count:2d} | "
            f"{s.boundary_type[:30]}{fault_marker}"
        )
    if len(sessions) > 20:
        lines.append(f"    ... ({len(sessions) - 20} more sessions)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_files = {
        "IRPRP30821 (worst case — 36% MCE ON coverage)":
            "/mnt/user-data/uploads/30821.xlsx",
        "IRPRP43771 (good — 102% coverage)":
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260309_044_A_3771.xlsx",
        "IRPRP42012 (medium — 66% coverage)":
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260403_044_A_2012.xlsx",
        "IRPRP37571 (good — 96% coverage)":
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260214_046_A_7571.xlsx",
    }

    for label, path in test_files.items():
        if not Path(path).exists():
            continue
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        df = pd.read_excel(path, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
        df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

        vehicle = df["Vehicle Name"].iloc[0] if "Vehicle Name" in df.columns else Path(path).stem
        print(f"  Vehicle : {vehicle}")
        print(f"  Events  : {len(df)}")

        sessions = detect_sessions(df)
        print(session_coverage_report(sessions, df))

        # Verify: compare old MCE-ON-only count vs new hybrid count
        old_mce_on = df["Dist Text"].str.contains("Power on of MCE", na=False).sum()
        print(f"  Old MCE-ON-only session count : {old_mce_on}")
        print(f"  New hybrid session count      : {len(sessions)}")
        improvement = len(sessions) - old_mce_on
        print(f"  Sessions recovered            : {improvement}")
