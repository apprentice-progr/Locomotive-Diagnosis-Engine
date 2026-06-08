"""
DDS Cross-Session Chain Tracker
================================
Extends dds_chain_matcher.py with cross-session causal chain linking.

Changes in this version
------------------------
  - Sessionizer integration: uses dds_sessionizer.detect_sessions() instead of
    MCE-ON-only boundaries. Fixes the ON-to-ON gap calculation bug.
  - Confidence decay: chains without terminal events decay 15% per empty session.
    Below 0.15 confidence they are expired and collapsed into a persistence record.
  - Hard age limit: non-deterministic chains with terminal events expire after 8h
    if no terminal found. Stops chains staying open across overnight stops.
  - Session type weighting: triggers in IDLE sessions start at 0.2x confidence.
  - Persistence collapsing: multiple DECAYED/EXPIRED fire/MR/OHE entries for the
    same chain_id are collapsed into one persistence record in the summary.
  - New outcome codes: DECAYED (confidence decayed out), PERSISTING (collapsed).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVELOPMENT ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 3 — Session-level anomaly baseline  [NOT STARTED]
STEP 4 — ATIL register integration       [NOT STARTED]
STEP 5 — ECode/EnvBl board localisation  [INTEGRATED]
STEP 6 — Cross-fleet pattern surfacing   [NOT STARTED]

See SYSTEM_DESIGN_AND_FINDINGS.md for full spec of each step.

CURRENT STATUS: Steps 1+2 complete. Sessionizer integrated. Decay added.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from dds_chain_matcher import (
    CHAIN_LIBRARY,
    match_chains,
    classify_session,
    check_persistence,
    analyse_deterministic_timing,
    _matches_any,
)
from dds_sessionizer import detect_sessions, session_coverage_report, Session

# Board localiser — optional import so the file still runs without it
try:
    from dds_board_localizer import annotate_chains as _annotate_chains
    BOARD_LOCALISER_AVAILABLE = True
except ImportError:
    BOARD_LOCALISER_AVAILABLE = False


# ATIL evidence engine — optional import
try:
    from atil_evidence_engine import ATILEvidenceEngine as _ATILEngine
    _atil_engine = _ATILEngine()
    ATIL_ENGINE_AVAILABLE = True
except ImportError:
    ATIL_ENGINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CROSS_SESSION_GAP_MAX_MIN = 120

# These chains override the gap check — deterministic timing IS the link evidence
DETERMINISTIC_CHAINS = {"IGBT_FEEDBACK"}

# Hard age limit (hours) for non-deterministic chains that DO have a terminal.
# After this long with no terminal found, the chain is expired regardless of gap.
# Calibrated from data: coolant→FPGA resolves within 60min, BUR within minutes.
# 8h is generous — covers any realistic cross-session causal window.
CHAIN_AGE_HARD_LIMIT_HRS = 8.0

# Chains without terminal events (fire, MR pressure, OHE) use decay instead of
# hard expiry because they represent persistent conditions, not time-bounded events.
# Decay factor per empty session (no propagation or terminal evidence found).
# 0.85^5 = 0.44, 0.85^10 = 0.20, 0.85^13 ≈ 0.15 → expiry threshold
DECAY_FACTOR          = 0.85
DECAY_MIN_CONFIDENCE  = 0.15   # below this → close as DECAYED

# Session type → base confidence multiplier for new chain triggers.
# A trigger in IDLE session is almost certainly a persistence fault, not a real event.
SESSION_CONFIDENCE_WEIGHT = {
    "FAULT_ACTIVE":    1.0,
    "INVESTIGATION":   1.0,
    "OPERATIONAL":     0.8,
    "TEST":            0.6,
    "IDLE":            0.2,
    "INCOMPLETE":      0.7,
    "UNKNOWN":         0.7,
}

FAULT_DRIVEN_MCE_OFF_SIGNALS = [
    "SS01 main power off", "SS02 traction bogie1 off", "SS03 traction bogie2 off",
    "Subsystem 02 & 03 off", "FPGA caused PS", "Protective shutdown",
    "Fast shutdown", "Soft shutdown", "Charging disabled, resistor too hot",
    "DC link voltage", "Inverter fault", "Isolation demand",
]


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class OpenChain:
    chain_id:         str
    chain_def:        dict
    trigger_text:     str
    trigger_time:     pd.Timestamp
    trigger_session:  int
    prop_hits:        List[dict] = field(default_factory=list)
    term_hits:        List[dict] = field(default_factory=list)
    dcu_origin:       Optional[str] = None
    confidence:       float = 0.3
    sessions_spanned: List[int] = field(default_factory=list)
    is_deterministic: bool = False
    is_intermittent:  bool = False
    closed:           bool = False
    close_reason:     str = ""
    session_type: Optional[str] = None
    # Decay tracking
    empty_sessions_since_evidence: int = 0
    age_hours:                     float = 0.0
    _uid: int = field(default_factory=lambda: id(object()))


@dataclass
class LinkedChainResult:
    chain_id:         str
    name:             str
    subsystem:        str
    severity:         str
    trigger_text:     str
    trigger_time:     pd.Timestamp
    trigger_session:  int
    terminal_time:    Optional[pd.Timestamp]
    terminal_text:    Optional[str]
    terminal_session: Optional[int]
    sessions_spanned: List[int]
    prop_hits:        List[dict]
    term_hits:        List[dict]
    confidence:       float
    dcu_origin:       Optional[str]
    is_deterministic: bool
    is_intermittent:  bool
    close_reason:     str
    action:           str
    cross_session:    bool
    # Outcome codes:
    #   SAME_SESSION  — trigger and terminal in same session
    #   CONFIRMED     — terminal/same-trigger/propagation found in later session
    #   EXPIRED       — gap > 120min or hard age limit hit
    #   CLEAN_OFF     — previous session ended cleanly, no causal link
    #   INCOMPLETE    — end of log, no terminal confirmed
    #   DECAYED       — confidence decayed below threshold (no-terminal chains)
    #   PERSISTING    — collapsed persistence record (multiple decayed instances)
    outcome:          str = "SAME_SESSION"
    # For PERSISTING records only
    persistence_count:  int = 0
    persistence_days:   float = 0.0
    # Board localisation — populated by annotate_chains() after tracker runs
    board_id:         Optional[str] = None   # e.g. "DCUM2_2"
    board_description: Optional[str] = None  # e.g. "Motor Converter 2 Control — Bogie 2"
    board_replaces:   Optional[str] = None   # e.g. "CON2-A607-A01 (DCU2/M2)"
    board_confidence: float = 0.0
    pcb_verdict:      Optional[str] = None   # CONFIRMED_CARD / PROBABLE_CARD / etc.
    pcb_score:        int = 0
    pcb_recommendation: Optional[str] = None
    atil_evidence: Optional[object] = None
    session_type: Optional[str] = None
    _uid:             int = field(default_factory=lambda: id(object()))


# ---------------------------------------------------------------------------
# FAULT-DRIVEN MCE OFF DETECTOR
# (kept for use when sessionizer is not available / fallback)
# ---------------------------------------------------------------------------

def is_fault_driven_mce_off(
    session_df: pd.DataFrame, lookback_min: float = 5.0
) -> Tuple[bool, str]:
    if session_df.empty:
        return False, "empty session"

    t_end      = session_df["Start time"].max()
    t_lookback = t_end - pd.Timedelta(minutes=lookback_min)
    tail       = session_df[session_df["Start time"] >= t_lookback]

    p1p2 = tail[tail["Prio"].isin([1, 2])]
    if not p1p2.empty:
        top = p1p2.iloc[-1]["Dist Text"]
        return True, f"P{int(p1p2.iloc[-1]['Prio'])} fault before end: '{str(top)[:60]}'"

    for sig in FAULT_DRIVEN_MCE_OFF_SIGNALS:
        if tail["Dist Text"].str.contains(sig, na=False, case=False).any():
            return True, f"Fault signal before end: '{sig}'"

    return False, "clean MCE off — no fault signals in final 5 min"


# ---------------------------------------------------------------------------
# CROSS-SESSION CHAIN TRACKER
# ---------------------------------------------------------------------------

class CrossSessionTracker:

    def __init__(self, gap_max_min: float = CROSS_SESSION_GAP_MAX_MIN):
        self.gap_max_min    = gap_max_min
        self.open_chains:   List[OpenChain] = []
        self.closed_chains: List[LinkedChainResult] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_result(
        self, oc: OpenChain, cross_session: bool, outcome: str
    ) -> LinkedChainResult:
        t_term  = oc.term_hits[-1]["time"]        if oc.term_hits else None
        tx_term = oc.term_hits[-1]["text"]        if oc.term_hits else None
        ts_term = oc.term_hits[-1].get("session") if oc.term_hits else None

        return LinkedChainResult(
            chain_id         = oc.chain_id,
            name             = oc.chain_def.get("name", oc.chain_id),
            subsystem        = oc.chain_def.get("subsystem", ""),
            severity         = oc.chain_def.get("severity", "MEDIUM"),
            trigger_text     = oc.trigger_text,
            trigger_time     = oc.trigger_time,
            trigger_session  = oc.trigger_session,
            terminal_time    = t_term,
            terminal_text    = tx_term,
            terminal_session = ts_term,
            sessions_spanned = list(oc.sessions_spanned),
            prop_hits        = list(oc.prop_hits),
            term_hits        = list(oc.term_hits),
            confidence       = round(oc.confidence, 2),
            dcu_origin       = oc.dcu_origin,
            is_deterministic = oc.is_deterministic,
            is_intermittent  = oc.is_intermittent,
            close_reason     = oc.close_reason,
            action           = oc.chain_def.get("action", ""),
            cross_session    = cross_session,
            outcome          = outcome,
            session_type = oc.session_type,
            _uid             = oc._uid,
        )

    def _check_propagation(self, oc: OpenChain, session_df: pd.DataFrame,
                           session_n: int) -> bool:
        found = False
        for cond in oc.chain_def.get("propagation", []):
            for _, row in session_df.iterrows():
                if _matches_any(row, [cond]):
                    lag = (row["Start time"] - oc.trigger_time).total_seconds() / 60
                    oc.prop_hits.append({
                        "text":    row["Dist Text"],
                        "time":    row["Start time"],
                        "lag_min": round(lag, 1),
                        "session": session_n,
                    })
                    oc.confidence = min(oc.confidence + 0.2, 1.0)
                    found = True
                    break
        return found

    def _check_terminal(self, oc: OpenChain, session_df: pd.DataFrame,
                        session_n: int) -> bool:
        found = False
        for cond in oc.chain_def.get("terminal", []):
            for _, row in session_df.iterrows():
                if _matches_any(row, [cond]):
                    lag = (row["Start time"] - oc.trigger_time).total_seconds() / 60
                    oc.term_hits.append({
                        "text":    row["Dist Text"],
                        "time":    row["Start time"],
                        "lag_min": round(lag, 1),
                        "session": session_n,
                    })
                    oc.confidence = min(oc.confidence + 0.25, 1.0)
                    found = True
                    break
        return found

    def _same_trigger_present(self, oc: OpenChain, session_df: pd.DataFrame) -> bool:
        search = oc.trigger_text[:40]
        return session_df["Dist Text"].str.contains(
            search, na=False, case=False, regex=False
        ).any()

    def _has_cross_session_propagation(self, oc: OpenChain) -> bool:
        return any(
            p.get("session", oc.trigger_session) != oc.trigger_session
            for p in oc.prop_hits
        )

    # ------------------------------------------------------------------
    # Main session processor
    # ------------------------------------------------------------------

    def process_session(
        self,
        session_n:         int,
        session_df:        pd.DataFrame,
        gap_from_prev_min: Optional[float],
        prev_session_df:   Optional[pd.DataFrame],
        session_type:      str = "UNKNOWN",
        is_fault_driven_prev_end: Optional[bool] = None,
    ):
        """
        Parameters
        ----------
        session_n             : 1-based session number
        session_df            : events in this session
        gap_from_prev_min     : actual OFF-to-ON gap (from sessionizer) — NOT ON-to-ON
        prev_session_df       : events in previous session (used if is_fault_driven_prev_end
                                is not provided by the sessionizer)
        session_type          : session type from sessionizer (OPERATIONAL/IDLE/etc.)
        is_fault_driven_prev_end : pre-computed from sessionizer; if None, computed here
        """
        if session_df is None or session_df.empty:
            return

        still_open: List[OpenChain] = []

        # ---- Step 1: Continue open chains from previous sessions ----

        for oc in self.open_chains:
            if session_n not in oc.sessions_spanned:
                oc.sessions_spanned.append(session_n)

            # Update age
            if not session_df.empty:
                oc.age_hours = (
                    session_df["Start time"].min() - oc.trigger_time
                ).total_seconds() / 3600

            # --- Gap check ---
            gap_ok = (
                gap_from_prev_min is not None
                and gap_from_prev_min <= self.gap_max_min
            )
            if oc.chain_id in DETERMINISTIC_CHAINS:
                gap_ok = True  # deterministic chains override gap limit

            if not gap_ok:
                oc.closed       = True
                oc.close_reason = (
                    f"Gap {gap_from_prev_min:.0f}min > {self.gap_max_min}min — "
                    f"sessions no longer causally linked"
                )
                self.closed_chains.append(
                    self._make_result(oc, cross_session=True, outcome="EXPIRED")
                )
                continue

            # --- Hard age limit for chains that DO have terminal conditions ---
            # No-terminal chains (fire, MR, OHE) use decay instead — see below.
            chain_has_terminal = bool(oc.chain_def.get("terminal"))
            if (
                chain_has_terminal
                and oc.chain_id not in DETERMINISTIC_CHAINS
                and oc.age_hours > CHAIN_AGE_HARD_LIMIT_HRS
            ):
                oc.closed       = True
                oc.close_reason = (
                    f"Chain aged out ({oc.age_hours:.1f}h > {CHAIN_AGE_HARD_LIMIT_HRS}h "
                    f"with no terminal found)"
                )
                self.closed_chains.append(
                    self._make_result(oc, cross_session=True, outcome="EXPIRED")
                )
                continue

            # --- Fault-driven end check ---
            # Use pre-computed value from sessionizer if available; otherwise compute here
            if is_fault_driven_prev_end is not None:
                fault_driven = is_fault_driven_prev_end
                fd_evidence  = "from sessionizer"
            elif prev_session_df is not None:
                fault_driven, fd_evidence = is_fault_driven_mce_off(prev_session_df)
            else:
                fault_driven, fd_evidence = True, "first session"

            if not fault_driven and oc.chain_id not in DETERMINISTIC_CHAINS:
                oc.closed       = True
                oc.close_reason = f"Clean MCE off before restart: {fd_evidence}"
                self.closed_chains.append(
                    self._make_result(oc, cross_session=True, outcome="CLEAN_OFF")
                )
                continue

            # --- Look for evidence in this session ---
            prop_found   = self._check_propagation(oc, session_df, session_n)
            term_found   = self._check_terminal(oc, session_df, session_n)
            same_trigger = self._same_trigger_present(oc, session_df)

            evidence_found = prop_found or term_found or same_trigger

            if term_found:
                oc.closed       = True
                oc.close_reason = (
                    f"Terminal event found in session {session_n} "
                    f"({gap_from_prev_min:.0f}min gap from previous)"
                )
                self.closed_chains.append(
                    self._make_result(oc, cross_session=True, outcome="CONFIRMED")
                )

            elif oc.chain_id in DETERMINISTIC_CHAINS and same_trigger:
                oc.confidence = min(oc.confidence + 0.15, 1.0)
                oc.empty_sessions_since_evidence = 0
                still_open.append(oc)

            elif same_trigger:
                oc.confidence   = min(oc.confidence + 0.1, 1.0)
                oc.closed       = True
                oc.close_reason = (
                    f"Same trigger still firing in session {session_n} — "
                    f"hardware not cleared by reset"
                )
                self.closed_chains.append(
                    self._make_result(oc, cross_session=True, outcome="CONFIRMED")
                )

            else:
                # No evidence found in this session.
                if evidence_found:
                    # prop_found but no terminal/same_trigger — reset empty counter
                    oc.empty_sessions_since_evidence = 0
                else:
                    oc.empty_sessions_since_evidence += 1

                # Decay for no-terminal chains
                if not chain_has_terminal:
                    oc.confidence *= DECAY_FACTOR ** max(
                        0, oc.empty_sessions_since_evidence - 1
                    )
                    oc.confidence = round(max(oc.confidence, 0.01), 3)

                    if oc.confidence < DECAY_MIN_CONFIDENCE:
                        oc.closed       = True
                        oc.close_reason = (
                            f"Confidence decayed to {oc.confidence:.2f} after "
                            f"{oc.empty_sessions_since_evidence} sessions without evidence"
                        )
                        self.closed_chains.append(
                            self._make_result(oc, cross_session=True, outcome="DECAYED")
                        )
                        continue

                still_open.append(oc)

        self.open_chains = still_open

        # ---- Step 2: Match new chains starting in this session ----

        new_matches = match_chains(session_df, CHAIN_LIBRARY)

        for m in new_matches:
            chain_def = next(
                (c for c in CHAIN_LIBRARY if c["chain_id"] == m["chain_id"]), {}
            )

            # Apply session type weighting to starting confidence
            sess_weight   = SESSION_CONFIDENCE_WEIGHT.get(session_type, 0.7)
            base_conf     = m["confidence"] * sess_weight

            if m["terminal_hits"]:
                # Both trigger and terminal in same session — close immediately
                for th in m["terminal_hits"]:
                    th.setdefault("session", session_n)
                cr = LinkedChainResult(
                    chain_id         = m["chain_id"],
                    name             = m["name"],
                    subsystem        = m["subsystem"],
                    severity         = m["severity"],
                    trigger_text     = m["trigger_text"],
                    trigger_time     = m["trigger_time"],
                    trigger_session  = session_n,
                    terminal_time    = m["terminal_hits"][-1]["time"],
                    terminal_text    = m["terminal_hits"][-1]["text"],
                    terminal_session = session_n,
                    sessions_spanned = [session_n],
                    prop_hits        = m["propagation_hits"],
                    term_hits        = m["terminal_hits"],
                    confidence       = round(base_conf, 2),
                    dcu_origin       = m["dcu_origin"],
                    is_deterministic = m["is_deterministic"],
                    is_intermittent  = m["is_intermittent"],
                    close_reason     = "Trigger and terminal in same session",
                    action           = chain_def.get("action", ""),
                    cross_session    = False,
                    outcome          = "SAME_SESSION",
                )
                self.closed_chains.append(cr)

            else:
                # Trigger only — open for cross-session tracking
                for ph in m["propagation_hits"]:
                    ph.setdefault("session", session_n)

                oc = OpenChain(
                    chain_id         = m["chain_id"],
                    chain_def        = chain_def,
                    trigger_text     = m["trigger_text"],
                    trigger_time     = m["trigger_time"],
                    trigger_session  = session_n,
                    prop_hits        = m["propagation_hits"],
                    dcu_origin       = m["dcu_origin"],
                    confidence       = round(base_conf, 2),
                    is_deterministic = m["is_deterministic"],
                    is_intermittent  = m["is_intermittent"],
                    sessions_spanned = [session_n],
                    session_type = session_type,
                )
                self.open_chains.append(oc)

    def flush_remaining(self):
        """Close any still-open chains at end of file."""
        for oc in self.open_chains:
            oc.closed       = True
            oc.close_reason = "End of log — no terminal event confirmed"
            if self._has_cross_session_propagation(oc):
                outcome = "CONFIRMED"
            elif oc.confidence < DECAY_MIN_CONFIDENCE:
                outcome = "DECAYED"
            else:
                outcome = "INCOMPLETE"
            self.closed_chains.append(
                self._make_result(
                    oc,
                    cross_session=len(oc.sessions_spanned) > 1,
                    outcome=outcome,
                )
            )
        self.open_chains = []


# ---------------------------------------------------------------------------
# DEDUPLICATION
# ---------------------------------------------------------------------------

def deduplicate_results(results: List[LinkedChainResult]) -> List[LinkedChainResult]:
    """Standard dedup: for same (chain_id, trigger_time), keep the best result."""
    best: Dict[str, LinkedChainResult] = {}

    for r in results:
        key = f"{r.chain_id}|{r.trigger_time.floor('s')}"

        if key not in best:
            best[key] = r
        else:
            existing = best[key]
            if (
                len(r.sessions_spanned) > len(existing.sessions_spanned)
                or (
                    len(r.sessions_spanned) == len(existing.sessions_spanned)
                    and r.confidence > existing.confidence
                )
            ):
                best[key] = r

    return list(best.values())


# Chains where CONFIRMED = "same trigger still firing" with no terminal
# are recurring faults, not distinct incidents.
# Collapse them if they span > this many days with no terminal found.
PERSIST_COLLAPSE_DAYS = 3

def collapse_persistence(results: List[LinkedChainResult]) -> List[LinkedChainResult]:
    """
    Two-pass collapsing:

    Pass 1 (noise collapse): DECAYED/EXPIRED/CLEAN_OFF entries for chains
    without any CONFIRMED results → single PERSISTING record.

    Pass 2 (confirmed persistence): CONFIRMED entries where ALL confirmed
    instances have no terminal event (close_reason contains "same trigger"
    or "hardware not cleared") AND span > PERSIST_COLLAPSE_DAYS → also
    collapse into PERSISTING. This handles MR pressure, pantograph repeat,
    OHE repeat patterns where the fault just keeps firing across sessions.
    Confirmed entries WITH actual terminal events are always kept individually.
    """
    out: List[LinkedChainResult] = []

    by_chain: Dict[str, List[LinkedChainResult]] = defaultdict(list)
    for r in results:
        by_chain[r.chain_id].append(r)

    for chain_id, hits in by_chain.items():
        confirmed   = [h for h in hits if h.outcome == "CONFIRMED"]
        same_sess   = [h for h in hits if h.outcome == "SAME_SESSION"]
        noise       = [h for h in hits if h.outcome in ("DECAYED", "EXPIRED", "CLEAN_OFF")]
        incomplete  = [h for h in hits if h.outcome == "INCOMPLETE"]

        out.extend(same_sess)
        out.extend(incomplete)

        # --- Pass 1: noise collapse ---
        if noise and not confirmed:
            first  = min(noise, key=lambda x: x.trigger_time)
            last   = max(noise, key=lambda x: x.trigger_time)
            p_days = (last.trigger_time - first.trigger_time).total_seconds() / 86400
            collapsed = LinkedChainResult(**{
                f.name: getattr(first, f.name)
                for f in first.__dataclass_fields__.values()
            })
            collapsed.outcome           = "PERSISTING"
            collapsed.persistence_count = len(noise)
            collapsed.persistence_days  = round(p_days, 1)
            collapsed.close_reason      = (
                f"Persistence record: {len(noise)} instances over {p_days:.0f} days"
            )
            collapsed.confidence        = round(max(h.confidence for h in noise), 2)
            out.append(collapsed)
            continue   # noise-only chain: done

        if noise and confirmed:
            # Confirmed chain with some noise — drop noise, keep confirmed
            pass  # noise discarded, confirmed handled below

        if not confirmed:
            continue

        # --- Pass 2: confirmed-persistence collapse ---
        # Split confirmed into: those with real terminals vs "same trigger" repeats
        with_terminal    = [h for h in confirmed if h.terminal_text is not None]
        without_terminal = [h for h in confirmed if h.terminal_text is None]

        # Always keep confirmed entries that have actual terminals
        out.extend(with_terminal)

        if not without_terminal:
            continue

        # Check if the no-terminal confirmed entries span > threshold
        first_nt = min(without_terminal, key=lambda x: x.trigger_time)
        last_nt  = max(without_terminal, key=lambda x: x.trigger_time)
        span_days = (last_nt.trigger_time - first_nt.trigger_time).total_seconds() / 86400

        if span_days > PERSIST_COLLAPSE_DAYS and len(without_terminal) > 2:
            # Collapse into one persistent record
            # Use the best-confidence instance as the representative
            best = max(without_terminal, key=lambda x: x.confidence)
            collapsed = LinkedChainResult(**{
                f.name: getattr(best, f.name)
                for f in best.__dataclass_fields__.values()
            })
            collapsed.outcome           = "PERSISTING"
            collapsed.persistence_count = len(without_terminal)
            collapsed.persistence_days  = round(span_days, 1)
            collapsed.close_reason      = (
                f"Recurring fault: {len(without_terminal)} instances over "
                f"{span_days:.0f} days, trigger repeating across sessions, "
                f"no terminal event ever confirmed — unresolved hardware fault"
            )
            collapsed.confidence        = round(
                max(h.confidence for h in without_terminal), 2
            )
            out.append(collapsed)
        else:
            # Short span or few instances — keep them individually
            out.extend(without_terminal)

    return out


# ---------------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------------

def _is_confirmed_cross(r: LinkedChainResult) -> bool:
    if not r.cross_session or len(r.sessions_spanned) < 2:
        return False
    return r.outcome == "CONFIRMED"


# ---------------------------------------------------------------------------
# FILE PROCESSOR
# ---------------------------------------------------------------------------

def process_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n{'='*70}")
    print(f"  DDS Cross-Session Chain Tracker — {path.name}")
    print(f"{'='*70}")

    df = pd.read_excel(file_path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
    df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

    vehicle   = df["Vehicle Name"].iloc[0] if "Vehicle Name" in df.columns else path.stem
    date_min  = df["Start time"].min().strftime("%d %b %Y")
    date_max  = df["Start time"].max().strftime("%d %b %Y")
    span_days = (df["Start time"].max() - df["Start time"].min()).days

    print(f"  Vehicle : {vehicle}")
    print(f"  Period  : {date_min} to {date_max}  ({span_days} days)")
    print(f"  Events  : {len(df)}  |  "
          f"P1: {(df['Prio']==1).sum()}  |  P2: {(df['Prio']==2).sum()}")

    # ---- Sessionizer (replaces old MCE-ON-only session builder) ----
    sessions_list: List[Session] = detect_sessions(df)
    print(session_coverage_report(sessions_list, df))

    # File-level analyses (use full df, not per-session)
    fire_chain = next(
        (c for c in CHAIN_LIBRARY if c["chain_id"] == "FIRE_DETECT_PERSISTENT"), None
    )
    fire_days, fire_first, fire_last = 0, None, None
    if fire_chain:
        fire_days, fire_first, fire_last = check_persistence(df, fire_chain)

    # IGBT analysis needs MCE ON times — pull from sessionizer output
    mce_on_times = [s.start_time for s in sessions_list
                    if "MCE ON" in s.boundary_type]
    if not mce_on_times:
        mce_on_times = [df["Start time"].min()]
    igbt_lags = analyse_deterministic_timing(df, "IGBT", mce_on_times)

    # ---- Run tracker with sessionizer sessions ----
    tracker  = CrossSessionTracker()
    prev_sdf: Optional[pd.DataFrame] = None
    prev_session: Optional[Session]  = None

    for s in sessions_list:
        if s.df is None or s.df.empty:
            prev_sdf     = s.df
            prev_session = s
            continue

        # Use the sessionizer's pre-computed fault-driven-end flag for the
        # PREVIOUS session, rather than recomputing it from prev_session_df.
        fd_prev = prev_session.is_fault_driven_end if prev_session is not None else True

        tracker.process_session(
            session_n                 = s.session_n,
            session_df                = s.df,
            gap_from_prev_min         = s.gap_before_min,
            prev_session_df           = prev_sdf,
            session_type              = s.session_type,
            is_fault_driven_prev_end  = fd_prev,
        )
        prev_sdf     = s.df
        prev_session = s

    tracker.flush_remaining()

    # Dedup then collapse persistence noise
    results = deduplicate_results(tracker.closed_chains)
    results = collapse_persistence(results)

    # Board localisation — enrich each result with ECode/EnvBl/PCB evidence
    if BOARD_LOCALISER_AVAILABLE:
        annotations = _annotate_chains(results, df)
        for r in results:
            ann = annotations.get(r._uid)
            if not ann:
                continue
            tb       = ann.get("trigger_board")
            pcb      = ann.get("pcb_suspect")
            guidance = ann.get("graduated_guidance", "")
            if tb and tb.board_id != "UNKNOWN":
                r.board_id          = tb.board_id
                r.board_description = tb.description
                # Use graduated guidance as replaces field so output
                # shows appropriate precision instead of raw card number
                r.board_replaces    = guidance if guidance else tb.replaces
                r.board_confidence  = tb.confidence
            if pcb and pcb.verdict not in ("SYSTEM_FAULT",):
                r.pcb_verdict        = pcb.verdict
                r.pcb_score          = pcb.score
                r.pcb_recommendation = pcb.recommendation
            # ATIL historical evidence
            if ATIL_ENGINE_AVAILABLE and r.outcome not in ("EXPIRED", "DECAYED", "CLEAN_OFF"):
                event_texts = []
                if r.prop_hits:
                    event_texts = [p.get("text", "") for p in r.prop_hits]
                r.atil_evidence = _atil_engine.process_chain(
                    chain_id     = r.chain_id,
                    trigger_text = r.trigger_text,
                    event_texts  = event_texts,
                    pcb_verdict  = getattr(r, "pcb_verdict", None),
                )

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    results.sort(key=lambda x: (sev_order.get(x.severity, 9), x.trigger_time))

    by_chain: Dict[str, List[LinkedChainResult]] = defaultdict(list)
    for r in results:
        by_chain[r.chain_id].append(r)

    # Outcome breakdown counts
    confirmed_xs = sum(1 for r in results if _is_confirmed_cross(r))
    single_sess  = sum(1 for r in results if r.outcome == "SAME_SESSION")
    incomplete   = sum(1 for r in results if r.outcome == "INCOMPLETE" and r.cross_session)
    persisting   = sum(1 for r in results if r.outcome == "PERSISTING")
    expired      = sum(1 for r in results if r.outcome in ("EXPIRED", "DECAYED", "CLEAN_OFF"))

    print(f"\n{'─'*70}")
    print(f"  CHAIN SUMMARY  ({len(results)} resolved chains)")
    print(f"  Single-session resolved  : {single_sess}")
    print(f"  Cross-session confirmed  : {confirmed_xs}  "
          f"(terminal / same-trigger / propagation)")
    print(f"  Cross-session incomplete : {incomplete}  (end of log)")
    print(f"  Persistence records      : {persisting}  (collapsed recurring faults)")
    print(f"  Suppressed               : {expired}  "
          f"(expired/decayed/clean-off — not actionable)")
    print(f"{'─'*70}")

    for chain_id, hits in sorted(
        by_chain.items(),
        key=lambda x: (sev_order.get(x[1][0].severity, 9), -len(x[1]))
    ):
        confirmed_hits  = [h for h in hits if _is_confirmed_cross(h)]
        single_hits     = [h for h in hits if h.outcome == "SAME_SESSION"]
        persist_hits    = [h for h in hits if h.outcome == "PERSISTING"]
        incomplete_hits = [h for h in hits if h.outcome == "INCOMPLETE" and h.cross_session]
        suppressed      = [h for h in hits
                           if h.outcome in ("EXPIRED", "DECAYED", "CLEAN_OFF")]

        avg_conf  = round(sum(h.confidence for h in hits) / len(hits), 2)
        best_conf = round(max(h.confidence for h in hits), 2)
        first_t   = min(h.trigger_time for h in hits)
        last_t    = max(h.trigger_time for h in hits)
        dcu_set   = {h.dcu_origin for h in hits if h.dcu_origin}
        max_sess  = max(len(h.sessions_spanned) for h in hits)

        print(f"\n  [{hits[0].severity}] {hits[0].name}")
        print(f"  Chain        : {chain_id}")
        print(f"  Total hits   : {len(hits)}  |  "
              f"confirmed x-session: {len(confirmed_hits)}  |  "
              f"single-session: {len(single_hits)}  |  "
              f"persistence: {len(persist_hits)}  |  "
              f"incomplete: {len(incomplete_hits)}  |  "
              f"suppressed: {len(suppressed)}")
        print(f"  Confidence   : avg={avg_conf}  best={best_conf}")
        print(f"  Span         : {first_t.strftime('%d %b %Y %H:%M')} "
              f"→ {last_t.strftime('%d %b %Y %H:%M')}")
        print(f"  Max sessions spanned: {max_sess}")
        if dcu_set:
            print(f"  DCU origin   : {', '.join(sorted(dcu_set))}")

        # Fire detection persistence
        if chain_id == "FIRE_DETECT_PERSISTENT" and fire_days > 0:
            crit_threshold = next(
                (c.get("persistence_critical_days", 7) for c in CHAIN_LIBRARY
                 if c["chain_id"] == "FIRE_DETECT_PERSISTENT"), 7
            )
            if fire_days >= crit_threshold:
                print(f"  *** FIRE DETECTION PERSISTENT: {fire_days} days "
                      f"({fire_first} → {fire_last})")
                print(f"      *** SAFETY WARNING: {fire_days} days > {crit_threshold}-day threshold."
                      f" Crew alarm fatigue risk. Loco unfit for line duty until certified clear.")
            else:
                print(f"  *** Fire alarm: {fire_days} day(s) seen "
                      f"({fire_first} → {fire_last}) — monitor, not yet safety-critical")

        # IGBT deterministic timing
        if chain_id == "IGBT_FEEDBACK" and igbt_lags:
            mean_lag = round(sum(igbt_lags) / len(igbt_lags), 1)
            std_lag  = round(float(np.std(igbt_lags)), 1)
            print(f"  *** DETERMINISTIC: {len(igbt_lags)} occurrences, "
                  f"mean {mean_lag}s after MCE ON (std={std_lag}s)")
            if std_lag < 30:
                print(f"      *** Hardware replacement required — "
                      f"fault timing consistent across all restart attempts")

        # Persistence record detail
        for ph in persist_hits:
            print(f"  *** RECURRING FAULT: {ph.persistence_count} instances over "
                  f"{ph.persistence_days:.0f} days "
                  f"(first {ph.trigger_time.strftime('%d %b')} — "
                  f"last {last_t.strftime('%d %b')})")

        # Confirmed cross-session links
        if confirmed_hits:
            print(f"  Confirmed cross-session links (showing up to 5):")
            for h in sorted(confirmed_hits, key=lambda x: x.trigger_time)[:5]:
                sess_str = "→".join(f"S{s}" for s in h.sessions_spanned)
                t_str    = (h.terminal_time.strftime("%d%b %H:%M")
                            if h.terminal_time else "no terminal confirmed")
                print(f"    {sess_str} | "
                      f"trigger {h.trigger_time.strftime('%d%b %H:%M')} | "
                      f"terminal {t_str} | conf={h.confidence}")
                if h.prop_hits:
                    cs_props = [p for p in h.prop_hits
                                if p.get("session", h.trigger_session) != h.trigger_session]
                    show = cs_props[0] if cs_props else (h.prop_hits[0] if h.prop_hits else None)
                    if show:
                        print(f"      prop:  {show['text'][:60]}")
                if h.terminal_text:
                    print(f"      term:  {h.terminal_text[:60]}")
                print(f"      close: {h.close_reason[:80]}")

        if any(h.is_intermittent for h in hits):
            print(f"  *** INTERMITTENT HARDWARE FLAG: "
                  f"PSPW/GPBPW w/o error cause present — "
                  f"inspect capacitors and connectors on line converter board")

        print(f"\n  Action: {hits[0].action}")

    # ---------------------------------------------------------------------------
    # CONFIRMED CROSS-SESSION INCIDENT DETAIL
    # ---------------------------------------------------------------------------
    confirmed_detail = [r for r in results if _is_confirmed_cross(r)]

    if confirmed_detail:
        print(f"\n{'─'*70}")
        print(f"  CONFIRMED CROSS-SESSION INCIDENTS  ({len(confirmed_detail)} incidents)")
        print(f"  (Expired / decayed / clean-off entries suppressed)")
        print(f"{'─'*70}")

        for r in sorted(
            confirmed_detail,
            key=lambda x: (sev_order.get(x.severity, 9), x.trigger_time)
        ):
            sess_str = " → ".join(f"S{s}" for s in r.sessions_spanned)
            span_min = (
                (r.terminal_time - r.trigger_time).total_seconds() / 60
                if r.terminal_time else None
            )
            print(f"\n  [{r.severity}] {r.name}")
            print(f"  Sessions : {sess_str}")
            print(f"  Trigger  : {r.trigger_time.strftime('%d %b %H:%M')}  "
                  f"S{r.trigger_session}  —  {str(r.trigger_text)[:65]}")
            if r.terminal_text:
                print(f"  Terminal : {r.terminal_time.strftime('%d %b %H:%M')}  "
                      f"S{r.terminal_session}  —  {str(r.terminal_text)[:65]}")
                if span_min is not None:
                    print(f"  Span     : {span_min:.0f} min "
                          f"({span_min/60:.1f}h) across {len(r.sessions_spanned)} sessions")
            if r.prop_hits:
                cs_props = [p for p in r.prop_hits
                            if p.get("session", r.trigger_session) != r.trigger_session]
                show_props = cs_props if cs_props else r.prop_hits
                print(f"  Propagation ({len(r.prop_hits)} event(s)"
                      f"{', ' + str(len(cs_props)) + ' cross-session' if cs_props else ''}):")
                for p in show_props[:3]:
                    sess_tag = f"S{p.get('session','?')}" if p.get("session") else ""
                    print(f"    {sess_tag} +{p['lag_min']}min  {str(p['text'])[:60]}")
            if r.dcu_origin:
                print(f"  DCU origin: {r.dcu_origin}")
            if r.is_deterministic:
                print(f"  *** DETERMINISTIC pattern confirmed")
            if r.is_intermittent:
                print(f"  *** INTERMITTENT hardware suspected")
            # Board localisation
            if r.board_id:
                conf_label = (
                    "high" if r.board_confidence >= 0.85 else
                    "medium" if r.board_confidence >= 0.70 else
                    "low"
                )
                print(f"  Board ({conf_label} conf): {r.board_description}")
                print(f"  Guidance   : {r.board_replaces}")
            if r.pcb_verdict and r.pcb_verdict != "SYSTEM_FAULT":
                print(f"  PCB verdict: {r.pcb_verdict}  (score={r.pcb_score})")
                if r.pcb_recommendation:
                    print(f"  PCB action : {r.pcb_recommendation[:100]}")
            print(f"  Confidence: {r.confidence}  |  Outcome: {r.outcome}"
                  f"  |  {r.close_reason[:65]}")
            if getattr(r, "atil_evidence", None) and r.atil_evidence.candidates:
                _atil_engine.print_result(r.atil_evidence, indent=2)
    else:
        print(f"\n  No confirmed cross-session incidents found.")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT — file dialog (multi-select) replaces CLI arguments
# ---------------------------------------------------------------------------

def select_files_via_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter not available. Pass file paths as command-line arguments.")
        return []

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    paths = filedialog.askopenfilenames(
        title     = "Select DDS Excel file(s) to analyse",
        filetypes = [
            ("Excel files", "*.xlsx *.xls *.xlsm"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_list = sys.argv[1:]
        print(f"Using {len(file_list)} file(s) from command-line arguments.")
    else:
        print("No files specified — opening file picker…")
        file_list = select_files_via_dialog()

    if not file_list:
        print("No files selected. Exiting.")
        sys.exit(0)

    print(f"\nFiles to process ({len(file_list)}):")
    for f in file_list:
        print(f"  {f}")

    for f in file_list:
        process_file(f)