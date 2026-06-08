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

def _sort_key(chain):
    sev = SEV_WEIGHT.get(getattr(chain, "severity", "HIGH").upper(), 3)
    t   = getattr(chain, "trigger_time", None) or pd.Timestamp.min
    return (sev, -int(t.timestamp()))

sorted_results = sorted(results, key=_sort_key)

high_chains   = [r for r in sorted_results if getattr(r, "severity", "").upper() == "HIGH"]
medium_chains = [r for r in sorted_results if getattr(r, "severity", "").upper() == "MEDIUM"]
persisting    = [r for r in sorted_results if getattr(r, "outcome", "") == "PERSISTING"]

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
        st.warning(
            f"**{len(high_chains)} HIGH priority chain{'s' if len(high_chains)>1 else ''}** · "
            f"{len(medium_chains)} MEDIUM"
        )
    if not results:
        st.success("No fault chains detected in this log.")

with hero_right:
    s1, s2, s3 = st.columns(3)
    s1.metric("P1 Events",  p1_count)
    s2.metric("P2 Events",  p2_count)
    s3.metric("Total",      f"{total_events:,}")
    st.caption(f"Sessions: {len(sessions_list)}  ·  Chains detected: {len(results)}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Shed Action Verdicts",
    "⏱️ Session Decomposition",
    "🔍 Raw Telemetry",
    "📄 Export Report",
])

with tab1:
    if not sorted_results:
        st.info("No fault chains detected.")
    else:
        for i, r in enumerate(sorted_results, 1):
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

            sev_icon    = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            board_tag   = f"→ {b_id}" if b_id not in ("UNKNOWN", "") else ""
            persist_tag = " 🔁 PERSISTING" if outcome == "PERSISTING" else ""
            label       = f"{sev_icon} {name}  {board_tag}{persist_tag}"

            with st.expander(label, expanded=(i <= 3)):
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

                    # Diagnostic procedure — always visible, muted style
                    action_text = getattr(r, "action", None) or getattr(r, "description", "")
                    if action_text:
                        st.divider()
                        st.caption("**Diagnostic & Shed Procedure:**")
                        st.caption(action_text)

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