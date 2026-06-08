"""
Integration test: dds_sessionizer → dds_cross_session
======================================================
Wires the new hybrid sessionizer into the cross-session tracker and
runs a before/after comparison on all four test locos.

Outputs per loco:
  - Session count: old vs new
  - Chain summary: old vs new (chain_id, hit count, cross-session count, best confidence)
  - Specific checks: IGBT, fire detection, VCB chains
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

# Import all three modules
from dds_sessionizer import detect_sessions, Session
from dds_cross_session import (
    CrossSessionTracker, LinkedChainResult,
    OpenChain, is_fault_driven_mce_off,
    CROSS_SESSION_GAP_MAX_MIN,
    DETERMINISTIC_CHAINS,
)
from dds_chain_matcher import (
    CHAIN_LIBRARY, match_chains,
    check_persistence, analyse_deterministic_timing,
    _matches_any,
)

# ---------------------------------------------------------------------------
# DEDUPLICATION  (extracted from process_file, now standalone)
# ---------------------------------------------------------------------------

def deduplicate(results: List[LinkedChainResult]) -> List[LinkedChainResult]:
    seen: Dict[str, LinkedChainResult] = {}
    out  = []
    for r in results:
        key = f"{r.chain_id}|{r.trigger_time.floor('s')}"
        if key not in seen:
            seen[key] = r
            out.append(r)
        else:
            ex = seen[key]
            if (len(r.sessions_spanned) > len(ex.sessions_spanned) or
                    r.confidence > ex.confidence):
                out.remove(ex)
                seen[key] = r
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# OLD METHOD: MCE-ON-only session builder (extracted from old process_file)
# ---------------------------------------------------------------------------

def old_build_sessions(df: pd.DataFrame):
    """
    Reproduces the original session-building logic from dds_cross_session.py
    so we can compare old vs new on the same data.
    Returns list of (session_n, sdf, duration_min, gap_from_prev_min).
    """
    mce_on_df    = df[df["Dist Text"].str.contains("Power on of MCE", na=False, case=False)]
    mce_on_times = mce_on_df["Start time"].tolist()
    if not mce_on_times:
        mce_on_times = [df["Start time"].min()]

    sessions = []
    for i, t_start in enumerate(mce_on_times):
        t_end = (mce_on_times[i + 1]
                 if i + 1 < len(mce_on_times)
                 else df["Start time"].max())
        sdf  = df[(df["Start time"] >= t_start) & (df["Start time"] < t_end)].copy()
        dur  = (t_end - t_start).total_seconds() / 60
        gap  = None if i == 0 else (
            t_start - mce_on_times[i - 1]).total_seconds() / 60
        sessions.append((i + 1, sdf, dur, gap))
    return sessions


# ---------------------------------------------------------------------------
# CHAIN RUNNER  (shared logic for both old and new)
# ---------------------------------------------------------------------------

def run_tracker(sessions_iter, df: pd.DataFrame) -> List[LinkedChainResult]:
    """
    Feed sessions through the CrossSessionTracker and return deduped results.
    sessions_iter: iterable of (session_n, sdf, duration_min, gap_from_prev_min)
                   OR list of Session objects from the new sessionizer.
    """
    tracker  = CrossSessionTracker()
    prev_sdf: Optional[pd.DataFrame] = None

    for item in sessions_iter:
        # Handle both old tuple format and new Session objects
        if isinstance(item, Session):
            sess_n      = item.session_n
            sdf         = item.df
            gap         = item.gap_before_min
            # Use pre-computed fault-driven-end from sessionizer
            # (overrides the tracker's own check for this session's prev)
        else:
            sess_n, sdf, _, gap = item

        if sdf is None or sdf.empty:
            prev_sdf = sdf if sdf is not None else prev_sdf
            continue

        tracker.process_session(sess_n, sdf, gap, prev_sdf)
        prev_sdf = sdf

    tracker.flush_remaining()
    return deduplicate(tracker.closed_chains)


# ---------------------------------------------------------------------------
# CHAIN SUMMARY  (compact for comparison)
# ---------------------------------------------------------------------------

def chain_summary(results: List[LinkedChainResult]) -> Dict:
    """
    Returns a dict: chain_id → {hits, cross, best_conf, max_sessions, severities}
    """
    by_chain = defaultdict(list)
    for r in results:
        by_chain[r.chain_id].append(r)

    summary = {}
    for cid, hits in by_chain.items():
        summary[cid] = {
            "hits":        len(hits),
            "cross":       sum(1 for h in hits if h.cross_session),
            "best_conf":   round(max(h.confidence for h in hits), 2),
            "max_sessions": max(len(h.sessions_spanned) for h in hits),
            "severity":    hits[0].severity,
            "first_trigger": min(h.trigger_time for h in hits),
            "last_trigger":  max(h.trigger_time for h in hits),
        }
    return summary


# ---------------------------------------------------------------------------
# SPECIFIC CHAIN CHECKS
# ---------------------------------------------------------------------------

def check_igbt(results: List[LinkedChainResult],
               df: pd.DataFrame,
               label: str):
    """Verify IGBT deterministic chain resolves correctly."""
    igbt = [r for r in results if r.chain_id == "IGBT_FEEDBACK"]
    if not igbt:
        print(f"  IGBT check [{label}]: NOT FOUND")
        return

    mce_on_times = df[
        df["Dist Text"].str.contains("Power on of MCE", na=False)
    ]["Start time"].tolist()
    lags = analyse_deterministic_timing(df, "IGBT", mce_on_times)
    mean_lag = round(np.mean(lags), 1) if lags else None
    std_lag  = round(float(np.std(lags)), 1) if lags else None

    best = max(igbt, key=lambda x: x.confidence)
    sess_str = "→".join(f"S{s}" for s in best.sessions_spanned)
    print(f"  IGBT check [{label}]:")
    print(f"    Occurrences: {len(igbt)}  best_conf={best.confidence}")
    print(f"    Best span: {sess_str}  ({len(best.sessions_spanned)} sessions)")
    print(f"    Deterministic: mean={mean_lag}s  std={std_lag}s  n={len(lags)}")
    if best.terminal_text:
        print(f"    Terminal confirmed: {best.terminal_text[:60]}")
    else:
        print(f"    Terminal: NOT confirmed")


def check_fire(results: List[LinkedChainResult],
               df: pd.DataFrame,
               label: str):
    """Verify fire detection persistence."""
    fire = [r for r in results if r.chain_id == "FIRE_DETECT_PERSISTENT"]
    fire_chain_def = next(
        (c for c in CHAIN_LIBRARY if c["chain_id"] == "FIRE_DETECT_PERSISTENT"), None
    )
    fire_days, fire_first, fire_last = 0, None, None
    if fire_chain_def:
        fire_days, fire_first, fire_last = check_persistence(df, fire_chain_def)

    print(f"  Fire check [{label}]:")
    print(f"    Chain instances: {len(fire)}  |  Persistence: {fire_days} days")
    if fire_days >= 7:
        print(f"    *** SAFETY: {fire_days} days persistent ({fire_first} → {fire_last})")
    # Key quality metric: fewer instances = better dedup/suppression
    cross = sum(1 for r in fire if r.cross_session)
    print(f"    Cross-session: {cross}  single-session: {len(fire)-cross}")


def check_vcb(results: List[LinkedChainResult],
              df: pd.DataFrame,
              label: str):
    """Verify VCB stuck chain."""
    vcb = [r for r in results if r.chain_id == "VCB_STUCK_ON"]
    raw_count = df["Dist Text"].str.contains("VCB will not open", na=False).sum()
    if not vcb:
        print(f"  VCB check [{label}]: chain NOT matched  "
              f"(raw event count: {raw_count})")
        return
    best = max(vcb, key=lambda x: x.confidence)
    cross = sum(1 for r in vcb if r.cross_session)
    print(f"  VCB check [{label}]:")
    print(f"    Raw events: {raw_count}  |  Chain instances: {len(vcb)}")
    print(f"    Best conf: {best.confidence}  cross-session: {cross}")
    print(f"    Best span: {'→'.join(f'S{s}' for s in best.sessions_spanned)}")


def check_coolant(results: List[LinkedChainResult],
                  df: pd.DataFrame,
                  label: str):
    """Verify coolant→FPGA chain with DCU origin."""
    cool = [r for r in results if r.chain_id == "COOLANT_FPGA"]
    raw_count = df["Dist Text"].str.contains(
        "Coolant pressure", na=False, case=False
    ).sum()
    if not cool:
        print(f"  Coolant check [{label}]: chain NOT matched  "
              f"(raw events: {raw_count})")
        return
    dcu_origins = {r.dcu_origin for r in cool if r.dcu_origin}
    cross = sum(1 for r in cool if r.cross_session)
    best  = max(cool, key=lambda x: x.confidence)
    print(f"  Coolant check [{label}]:")
    print(f"    Raw events: {raw_count}  |  Chain instances: {len(cool)}")
    print(f"    Best conf: {best.confidence}  cross-session: {cross}")
    print(f"    DCU origins identified: {dcu_origins or 'none'}")
    if best.prop_hits:
        print(f"    Best propagation: {best.prop_hits[0]['text'][:60]}")


# ---------------------------------------------------------------------------
# COMPARISON PRINTER
# ---------------------------------------------------------------------------

def print_comparison(loco: str,
                     old_summary: Dict, new_summary: Dict,
                     old_session_n: int, new_session_n: int):

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    print(f"\n  {'─'*64}")
    print(f"  CHAIN COMPARISON — {loco}")
    print(f"  Sessions:  old={old_session_n}  new={new_session_n}  "
          f"(+{new_session_n - old_session_n})")
    print(f"  {'─'*64}")

    all_chains = sorted(
        set(old_summary) | set(new_summary),
        key=lambda c: sev_order.get(
            (old_summary.get(c) or new_summary.get(c))["severity"], 9
        )
    )

    header = f"  {'Chain':<28} {'OLD':>5} {'NEW':>5}  {'Δhits':>6}  " \
             f"{'OLD_conf':>9} {'NEW_conf':>9}  {'OLD_cross':>10} {'NEW_cross':>10}"
    print(header)
    print(f"  {'─'*len(header.rstrip())}")

    for cid in all_chains:
        o = old_summary.get(cid)
        n = new_summary.get(cid)

        o_hits  = o["hits"]        if o else 0
        n_hits  = n["hits"]        if n else 0
        o_conf  = o["best_conf"]   if o else "-"
        n_conf  = n["best_conf"]   if n else "-"
        o_cross = o["cross"]       if o else 0
        n_cross = n["cross"]       if n else 0
        delta   = n_hits - o_hits

        # Flag interesting changes
        flag = ""
        if o and n:
            if n["best_conf"] > o["best_conf"] + 0.1:
                flag = " ↑conf"
            elif n["best_conf"] < o["best_conf"] - 0.1:
                flag = " ↓conf"
            if n["max_sessions"] > o["max_sessions"]:
                flag += f" span+{n['max_sessions']-o['max_sessions']}"

        delta_str = f"{delta:+d}" if delta != 0 else "  ="
        print(f"  {cid:<28} {o_hits:>5} {n_hits:>5}  {delta_str:>6}  "
              f"{str(o_conf):>9} {str(n_conf):>9}  "
              f"{o_cross:>10} {n_cross:>10}{flag}")

    print()


# ---------------------------------------------------------------------------
# MAIN COMPARISON RUNNER
# ---------------------------------------------------------------------------

TEST_FILES = {
    "IRPRP43771": "/mnt/user-data/uploads/ED_V_IR_PRP___20260309_044_A_3771.xlsx",
    "IRPRP42012": "/mnt/user-data/uploads/ED_V_IR_PRP___20260403_044_A_2012.xlsx",
    "IRPRP30821": "/mnt/user-data/uploads/30821.xlsx",
    "IRPRP37571": "/mnt/user-data/uploads/ED_V_IR_PRP___20260214_046_A_7571.xlsx",
}

def main():
    for loco, path in TEST_FILES.items():
        if not Path(path).exists():
            print(f"\nSkipping {loco} — file not found")
            continue

        print(f"\n{'='*70}")
        print(f"  {loco}")
        print(f"{'='*70}")

        df = pd.read_excel(path, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
        df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

        print(f"  Events: {len(df)}  "
              f"P1: {(df['Prio']==1).sum()}  "
              f"P2: {(df['Prio']==2).sum()}")

        # ---- OLD method ----
        old_sessions = old_build_sessions(df)
        old_results  = run_tracker(old_sessions, df)
        old_summary  = chain_summary(old_results)
        old_n        = len(old_sessions)

        # ---- NEW method ----
        new_sessions = detect_sessions(df)
        new_results  = run_tracker(new_sessions, df)
        new_summary  = chain_summary(new_results)
        new_n        = len(new_sessions)

        # Print comparison table
        print_comparison(loco, old_summary, new_summary, old_n, new_n)

        # Specific chain checks (new method only)
        print(f"  Specific chain verification (NEW sessionizer):")
        check_igbt(new_results, df, "NEW")
        check_fire(new_results, df, "NEW")
        check_vcb(new_results, df, "NEW")
        check_coolant(new_results, df, "NEW")

        # Session type breakdown
        type_counts = defaultdict(int)
        for s in new_sessions:
            type_counts[s.session_type] += 1
        print(f"\n  New session types: "
              f"{dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

        # Coverage check
        covered = sum(s.n_events for s in new_sessions)
        print(f"  Event coverage: {covered}/{len(df)} "
              f"({'100' if covered==len(df) else f'{covered/len(df)*100:.1f}'}%)")

    print(f"\n{'='*70}")
    print("  Integration test complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
