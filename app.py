import streamlit as st
import pandas as pd

from dds_sessionizer import detect_sessions, session_coverage_report
from dds_pdf_report import build_report
from dds_cross_session import CrossSessionTracker, deduplicate_results, collapse_persistence
from dds_board_localizer import annotate_chains
from atil_evidence_engine import ATILEvidenceEngine

_atil_engine = ATILEvidenceEngine()

st.set_page_config(
    page_title="DDS Fault Engine — ELS/BL",
    page_icon="🚂",
    layout="wide"
)

st.markdown("## Locomotive DDS Fault Analysis Engine")
st.caption("WAG-9 / WAP-5 / WAP-7 · MICAS-S2 · ELS/BL Shed · Upload a DDS Excel export to begin.")

uploaded_file = st.file_uploader(
    "Choose a DDS Excel File (.xlsx)",
    type=["xlsx"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.stop()

clock_fault = False
bad_year = None
with st.spinner("Running diagnostic pipeline…"):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        df.columns = [str(c).strip() for c in df.columns]
        df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
        df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

        MAX_DATE = pd.Timestamp("2028-01-01")
        clock_fault = df["Start time"].max() > MAX_DATE
        if clock_fault:
            bad_year = df["Start time"].max().year
            df = df[df["Start time"] <= MAX_DATE].reset_index(drop=True)

        vehicle      = str(df["Vehicle Name"].iloc[0]) if "Vehicle Name" in df.columns else "Unknown"
        total_events = len(df)
        p1_count     = int((df["Prio"] == 1).sum())
        p2_count     = int((df["Prio"] == 2).sum())
        log_start    = df["Start time"].min().strftime("%d %b %Y")
        log_end      = df["Start time"].max().strftime("%d %b %Y")

        sessions_list = detect_sessions(df)

        tracker   = CrossSessionTracker()
        prev_sdf  = None
        prev_sess = None
        for s in sessions_list:
            if s.df is None or s.df.empty:
                prev_sdf = s.df; prev_sess = s; continue
            fd_prev = prev_sess.is_fault_driven_end if prev_sess is not None else True
            tracker.process_session(
                session_n=s.session_n, session_df=s.df,
                gap_from_prev_min=s.gap_before_min, prev_session_df=prev_sdf,
                session_type=s.session_type, is_fault_driven_prev_end=fd_prev,
            )
            prev_sdf = s.df; prev_sess = s

        tracker.flush_remaining()
        results = collapse_persistence(deduplicate_results(tracker.closed_chains))
        localizer_annotations = annotate_chains(results, df)

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()

SEV_WEIGHT = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Sessions that represent unscheduled/testing activity rather than real line
# operation. A HIGH severity chain firing during one of these should not
# outrank a real fault from OPERATIONAL or FAULT_ACTIVE running.
TEST_SESSION_TYPES = set()  # was {"TEST", "IDLE"} — session_type is still recorded
                             # and visible per-chain; it no longer suppresses or
                             # demotes a chain that already matched on its own evidence.

def _sort_key(chain):
    sev = SEV_WEIGHT.get(getattr(chain, "severity", "HIGH").upper(), 3)
    is_test = getattr(chain, "session_type", "") in TEST_SESSION_TYPES
    # Demote test/idle-session faults below all real-session faults of the
    # same severity tier, but keep them above the next severity tier down
    # only as a tiebreak — i.e. (severity, is_test, recency).
    t = getattr(chain, "trigger_time", None) or pd.Timestamp.min
    return (sev, 1 if is_test else 0, -int(t.timestamp()))

sorted_results = sorted(results, key=_sort_key)

high_chains   = [r for r in sorted_results
                 if getattr(r, "severity", "").upper() == "HIGH"
                 and getattr(r, "session_type", "") not in TEST_SESSION_TYPES]
medium_chains = [r for r in sorted_results
                 if getattr(r, "severity", "").upper() == "MEDIUM"
                 and getattr(r, "session_type", "") not in TEST_SESSION_TYPES]
test_session_chains = [r for r in sorted_results
                        if getattr(r, "session_type", "") in TEST_SESSION_TYPES]
persisting    = [r for r in sorted_results if getattr(r, "outcome", "") == "PERSISTING"]

# ── Group occurrences by distinct fault type ─────────────────────────────────
# A single chain_id (e.g. PANTO_BOUNCE, HB1_MCB_CLUSTER) can have many
# occurrences across the log period — each a real, independently-verified
# recurrence, not noise (see dds_chain_matcher's burst-collapse fix). But a
# technician scanning the verdict list wants "is panto bounce a problem on
# this loco" once, not eight separate cards for the same fault type. So the
# main list shows one card per chain_id — the most recent occurrence — with
# every earlier occurrence still available, just folded underneath rather
# than competing for top-billing.
#
# Nothing about detection, counting, or confidence changes here: this is
# purely how sorted_results gets laid out on screen.

def _group_by_chain_id(chain_list):
    """
    Group a list of chain results by chain_id. Returns a list of dicts:
    {"chain_id", "representative": most-recent r, "all": [r, ...] sorted
    most-recent-first, "count": n}, sorted by (severity, recency of the
    representative) — i.e. the same ordering _sort_key already gives the
    flat list, just collapsed to one row per fault type.
    """
    groups = {}
    for r in chain_list:
        cid = getattr(r, "chain_id", getattr(r, "name", "UNKNOWN"))
        groups.setdefault(cid, []).append(r)

    grouped = []
    for cid, occurrences in groups.items():
        occurrences_sorted = sorted(
            occurrences,
            key=lambda r: getattr(r, "trigger_time", None) or pd.Timestamp.min,
            reverse=True,
        )
        grouped.append({
            "chain_id":       cid,
            "representative": occurrences_sorted[0],
            "all":            occurrences_sorted,
            "count":          len(occurrences_sorted),
        })

    grouped.sort(key=lambda g: _sort_key(g["representative"]))
    return grouped

grouped_results = _group_by_chain_id(sorted_results)


# Compute uncaptured P1 count here so hero stats can show it
_captured_keys = set()
for r in sorted_results:
    t = getattr(r, "trigger_text", "")
    if t:
        _captured_keys.add(t[:50])

_uncaptured_p1 = []
_seen_uc = set()
if "Prio" in df.columns and "Dist Text" in df.columns:
    for ev in df[df["Prio"] == 1]["Dist Text"].dropna().tolist():
        key = ev[:50]
        if key not in _captured_keys and key not in _seen_uc:
            if not any(key[:30] in c for c in _captured_keys):
                _seen_uc.add(key)
                _uncaptured_p1.append(ev)

if clock_fault:
    st.warning(
        f"⚠️ Clock anomaly on this loco: timestamps extend to {bad_year}. "
        f"Rows beyond 2028 excluded — check MCE system clock at next service."
    )

hero_left, hero_right = st.columns([3, 2])

with hero_left:
    st.markdown(f"### {vehicle}")
    st.caption(f"Log period: {log_start} → {log_end}  ·  {len(sessions_list)} operational sessions")

    if persisting:
        names = [getattr(r, "name", r.chain_id) for r in persisting[:3]]
        st.error(
            f"**{len(persisting)} fault{'s' if len(persisting)>1 else ''} persisting across sessions:** "
            + " · ".join(names)
            + ("…" if len(persisting) > 3 else "")
        )
    if high_chains:
        n_high_types = sum(1 for g in grouped_results if g["representative"] in high_chains)
        n_med_types  = sum(1 for g in grouped_results if g["representative"] in medium_chains)
        st.warning(
            f"**{n_high_types} HIGH priority fault type{'s' if n_high_types!=1 else ''}** "
            f"({len(high_chains)} occurrence{'s' if len(high_chains)!=1 else ''}) · "
            f"{n_med_types} MEDIUM"
        )
    if test_session_chains:
        st.info(
            f"ℹ️ {len(test_session_chains)} chain"
            f"{'s' if len(test_session_chains)>1 else ''} fired during TEST/IDLE "
            f"sessions (unscheduled inspection or workshop testing) — shown lower "
            f"in the list below, not counted as line-operation faults."
        )
    if not results:
        st.success("No fault chains detected in this log.")

with hero_right:
    s1, s2, s3 = st.columns(3)
    s1.metric("P1 Events",  p1_count)
    s2.metric("P2 Events",  p2_count)
    s3.metric("Total",      f"{total_events:,}")
    st.caption(
        f"Sessions: {len(sessions_list)}  ·  Chains detected: {len(results)}"
        + (f"  ·  ⚠️ {len(_uncaptured_p1)} uncaptured P1 type{'s' if len(_uncaptured_p1)!=1 else ''}" if _uncaptured_p1 else "")
    )

st.divider()

# ── Enrich uncaptured P1s with date/count info ───────────────────────────────
_uncaptured_p1_rich = []
if _uncaptured_p1 and "Dist Text" in df.columns:
    p1_df = df[df["Prio"] == 1].copy()
    for ev in _uncaptured_p1:
        matches = p1_df[p1_df["Dist Text"] == ev]
        if matches.empty:
            matches = p1_df[p1_df["Dist Text"].str.contains(ev[:40], na=False, regex=False)]
        if not matches.empty:
            times = matches["Start time"].dropna().sort_values()
            _uncaptured_p1_rich.append({
                "text":       ev,
                "count":      len(matches),
                "first":      times.iloc[0]  if len(times) > 0 else None,
                "last":       times.iloc[-1] if len(times) > 0 else None,
            })
    # Sort by most recent first
    _uncaptured_p1_rich.sort(
        key=lambda x: x["last"] if x["last"] is not None else pd.Timestamp.min,
        reverse=True
    )

uc_label = f"⚠️ Unmatched P1s ({len(_uncaptured_p1)})" if _uncaptured_p1 else "⚠️ Unmatched P1s"

# ── Uncaptured P1 card renderer — defined at module level, not inside tab ────
_UC_TOP_N    = 5
_log_end_ts  = df["Start time"].max()

def _render_uc_cards(items):
    for item in items:
        first_str = item["first"].strftime("%d-%b-%Y") if item["first"] else "—"
        last_str  = item["last"].strftime("%d-%b-%Y")  if item["last"]  else "—"
        is_recent = (
            item["last"] is not None and
            (_log_end_ts - item["last"]).days <= 14
        )
        badge = "🔴" if is_recent else "⚪"
        with st.expander(f"{badge} {item['text'][:80]}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Occurrences", item["count"])
            c2.metric("First seen",  first_str)
            c3.metric("Last seen",   last_str)
            if is_recent:
                st.caption("🔴 Active recently — check Raw Telemetry for full context.")
            else:
                st.caption("No matching fault chain defined. If this fault recurs, consider adding a chain.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔧 Shed Action Verdicts",
    "⏱️ Session Decomposition",
    "🔍 Raw Telemetry",
    "📄 Export Report",
    uc_label,
])

with tab1:
    if not sorted_results:
        st.info("No fault chains detected.")
    else:
        for i, g in enumerate(grouped_results, 1):
            r         = g["representative"]
            n_occ     = g["count"]
            earlier   = g["all"][1:]   # everything except the representative itself

            severity  = getattr(r, "severity", "HIGH").upper()
            chain_id  = getattr(r, "chain_id",  "UNKNOWN")
            name      = getattr(r, "name",      chain_id)
            outcome   = getattr(r, "outcome",   "UNKNOWN")

            if name == chain_id and hasattr(r, "trigger_text"):
                name = r.trigger_text.split("-")[-1].strip() if "-" in r.trigger_text else chain_id

            chain_uid = getattr(r, "_uid", None)
            ann       = localizer_annotations.get(chain_uid) if chain_uid else None
            tb        = ann.get("trigger_board")         if ann else None
            guidance  = ann.get("graduated_guidance", "") if ann else ""
            b_id      = getattr(tb, "board_id",    "UNKNOWN") if tb else "UNKNOWN"
            b_desc    = getattr(tb, "description", "")        if tb else ""
            b_rep     = getattr(tb, "replaces",    "")        if tb else ""

            sess_type   = getattr(r, "session_type", "")
            is_test_sess = sess_type in TEST_SESSION_TYPES

            sev_icon    = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            board_tag   = f"→ {b_id}" if b_id not in ("UNKNOWN", "") else ""
            persist_tag = " 🔁 PERSISTING" if outcome == "PERSISTING" else ""
            test_tag    = f" 🧪 {sess_type} SESSION" if is_test_sess else ""
            recur_tag   = f"  ·  ↻ {n_occ}× recorded" if n_occ > 1 else ""
            label       = f"{sev_icon} {name}  {board_tag}{persist_tag}{test_tag}{recur_tag}"

            with st.expander(label, expanded=(i <= 3 and not is_test_sess)):
                if n_occ > 1:
                    first_seen = g["all"][-1].trigger_time
                    last_seen  = g["all"][0].trigger_time
                    st.caption(
                        f"↻ **Recorded {n_occ} times** in this log "
                        f"({first_seen.strftime('%d-%b-%Y') if first_seen else '—'} "
                        f"→ {last_seen.strftime('%d-%b-%Y') if last_seen else '—'}). "
                        f"Showing most recent occurrence below; earlier ones listed at the bottom."
                    )
                if is_test_sess:
                    st.caption(
                        f"🧪 **This fault occurred during a {sess_type} session** "
                        f"(unscheduled inspection, workshop testing, or idle period) — "
                        f"not during normal line operation. Confidence reduced accordingly."
                    )

                action_col, context_col = st.columns([4, 4])

                atil = _atil_engine.process_chain(
                    chain_id     = chain_id,
                    trigger_text = getattr(r, "trigger_text", ""),
                    pcb_verdict  = getattr(
                        ann.get("pcb_suspect") if ann else None, "verdict", None),
                )

                with action_col:
                    # Board ID — largest element, first thing the eye goes to
                    if b_id not in ("UNKNOWN", ""):
                        st.markdown(
                            f"<div style='font-size:2rem;font-weight:700;line-height:1.1;"
                            f"letter-spacing:0.04em;margin-bottom:0.1rem'>{b_id}</div>"
                            f"<div style='font-size:0.95rem;color:#aaa;margin-bottom:0.6rem'>"
                            f"{b_desc}</div>",
                            unsafe_allow_html=True,
                        )
                        if b_rep and b_rep != "Physical inspection required":
                            st.caption(f"📍 {b_rep}")
                    else:
                        st.markdown(
                            "<div style='font-size:1.4rem;font-weight:600;color:#888'>"
                            "Inspect physically</div>",
                            unsafe_allow_html=True,
                        )

                    # ATIL fleet evidence bars
                    if atil.candidates:
                        top = atil.candidates[0]
                        filled = int(top.pct / 5)
                        bar = "█" * filled + "░" * (20 - filled)
                        st.markdown(
                            f"`{bar}` &nbsp;**{top.pct:.0f}%** — {top.name}",
                            unsafe_allow_html=True,
                        )
                        for c in atil.candidates[1:4]:
                            f2 = int(c.pct / 5)
                            st.caption(f"`{'█'*f2}{'░'*(20-f2)}` {c.pct:.0f}% — {c.name}")
                        st.caption(
                            f"Fleet evidence · "
                            f"{atil.evidence_lines} pattern"
                            f"{'s' if atil.evidence_lines != 1 else ''} · "
                            f"{atil.note}"
                        )

                    # Graduated guidance
                    if guidance:
                        st.markdown(f"**Action:** {guidance}")

                with context_col:
                    # Identity pills
                    mc = st.columns(3)
                    mc[0].markdown(f"**Chain**  \n`{chain_id}`")
                    mc[1].markdown(f"**Status**  \n`{outcome}`")
                    mc[2].markdown(f"**Priority**  \n`{severity}`")

                    if getattr(r, "trigger_time", None):
                        span_end = getattr(r, "span_end", None)
                        st.markdown(
                            f"**Trigger:** `{r.trigger_time.strftime('%d-%b-%Y  %H:%M:%S')}`"
                            + (f"  →  `{span_end}`" if span_end else "")
                        )
                    if getattr(r, "trigger_text", None):
                        st.markdown(f"**Root cause event:** `{r.trigger_text}`")

                    # ── Why this verdict? ─────────────────────────────────
                    # Shows the exact data fields that drove localisation —
                    # for presentation/defence: "the tool concluded X because..."
                    evidence   = ann.get("evidence", {}) if ann else {}
                    ev_ecode   = evidence.get("ecode",      "—")
                    ev_envbl   = evidence.get("envbl",      "—")
                    ev_evname  = evidence.get("event_name", "—")
                    ev_source  = evidence.get("source",     "")
                    if ev_source and ev_source not in ("—", "none", "no trigger row matched"):
                        st.divider()
                        st.caption("**Why this verdict?**")
                        st.caption(f"🔢 **ECode 0:** `{ev_ecode}`")
                        st.caption(f"🗂️ **EnvBl Id:** `{ev_envbl}`")
                        if ev_evname and ev_evname not in ("—", "nan", "None", ""):
                            st.caption(f"📋 **Event Name:** `{ev_evname}`")
                        st.caption(f"🧠 **Localisation basis:** _{ev_source}_")
                        st.caption(
                            "Source: Bombardier FFM 3EH-214057-0001 §6.3.3.9 "
                            "(processor disturbance forms) + observed DDS data"
                        )

                    # Diagnostic procedure — always visible, muted style
                    action_text = getattr(r, "action", None) or getattr(r, "description", "")
                    if action_text:
                        st.divider()
                        st.caption("**Diagnostic & Shed Procedure:**")
                        st.caption(action_text)

                # ── Earlier occurrences of this same fault type ───────────
                if earlier:
                    st.divider()
                    with st.expander(
                        f"↻ {len(earlier)} earlier occurrence{'s' if len(earlier)!=1 else ''} "
                        f"of this fault", expanded=False
                    ):
                        for er in earlier:
                            er_sess   = getattr(er, "session_type", "")
                            er_test   = " 🧪" if er_sess in TEST_SESSION_TYPES else ""
                            er_time   = getattr(er, "trigger_time", None)
                            er_time_s = er_time.strftime("%d-%b-%Y  %H:%M:%S") if er_time else "—"
                            er_out    = getattr(er, "outcome", "UNKNOWN")
                            st.caption(
                                f"`{er_time_s}`  ·  {er_out}  ·  conf {getattr(er, 'confidence', 0):.2f}{er_test}"
                            )

        # ── Uncaptured P1 pointer ─────────────────────────────────────
        if _uncaptured_p1:
            st.divider()
            st.caption(
                f"⚠️ **{len(_uncaptured_p1)} unique P1 event type{'s' if len(_uncaptured_p1)>1 else ''}** "
                f"fired but did not match any known fault chain — see the **{uc_label}** tab for details."
            )

with tab2:
    st.caption("Hybrid power-cycle session detection — MCE ON events + fault-driven boundaries")
    st.text(session_coverage_report(sessions_list, df))

    rows = []
    for s in sessions_list:
        rows.append({
            "Session":          s.session_n,
            "Type":             s.session_type,
            "Boundary":         s.boundary_type,
            "Start":            s.start_time,
            "Duration (min)":   round(s.duration_min, 1) if s.duration_min else 0.0,
            "Gap before (min)": round(s.gap_before_min, 1) if s.gap_before_min else 0.0,
            "Events":           s.n_events,
        })
    sdf = pd.DataFrame(rows)
    if not sdf.empty:
        sdf = sdf.sort_values("Start", ascending=False)
    st.dataframe(sdf, use_container_width=True)

with tab4:
    st.caption("Generate a PDF for records or handoff to crew / JE.")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Shed Action Report")
        st.caption(
            "Portrait · One section per fault chain · "
            "Card ID, fleet evidence, procedure. For crew briefing or daily record."
        )
        if st.button("Generate Action Report PDF", key="gen_action"):
            with st.spinner("Building PDF…"):
                pdf_bytes = build_report(
                    vehicle, log_start, log_end,
                    sorted_results, localizer_annotations, _atil_engine,
                    sessions_list, report_type="action",
                )
            fname = f"DDS_ActionReport_{vehicle}_{log_end.replace(' ', '')}.pdf"
            st.download_button(
                label="⬇ Download Action Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                key="dl_action",
            )

    with col_b:
        st.markdown("#### ATIL Failure Register (2026-27 Format)")
        st.caption(
            "Landscape · Columns mirror the official ATIL register sheet · "
            "S.N., Loco, Date, Equipment, Failed Item, Investigation. "
            "Ready to copy into the register or submit as-is."
        )
        if st.button("Generate ATIL Register PDF", key="gen_atil"):
            with st.spinner("Building PDF…"):
                pdf_bytes = build_report(
                    vehicle, log_start, log_end,
                    sorted_results, localizer_annotations, _atil_engine,
                    sessions_list, report_type="atil_register",
                )
            fname = f"ATIL_Register_{vehicle}_{log_end.replace(' ', '')}.pdf"
            st.download_button(
                label="⬇ Download ATIL Register",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                key="dl_atil",
            )

with tab3:
    st.caption("Reverse-chronological raw event log. Use column headers to filter.")
    cols = [c for c in
            ["Start time", "Event Name", "Dist Text", "ECode 0", "EnvBl Id", "Prio"]
            if c in df.columns]
    st.dataframe(
        df[cols].sort_values("Start time", ascending=False),
        use_container_width=True,
    )

with tab5:
    if not _uncaptured_p1_rich:
        st.info("No uncaptured P1 events — all P1 faults matched a known chain.")
    else:
        st.caption(
            f"{len(_uncaptured_p1_rich)} unique P1 event type{'s' if len(_uncaptured_p1_rich)>1 else ''} "
            f"fired but did not match any known fault chain. "
            f"Sorted by most recent occurrence. Use Raw Telemetry tab for full event detail."
        )
        _render_uc_cards(_uncaptured_p1_rich[:_UC_TOP_N])
        if len(_uncaptured_p1_rich) > _UC_TOP_N:
            with st.expander(f"Show remaining {len(_uncaptured_p1_rich) - _UC_TOP_N} events", expanded=False):
                _render_uc_cards(_uncaptured_p1_rich[_UC_TOP_N:])