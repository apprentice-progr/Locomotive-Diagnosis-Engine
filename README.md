# Locomotive DDS Fault Analysis Engine

Advanced cross-session diagnostics and FFM hardware board localisation for WAG-9/WAP-5/WAP-7 electric locomotives using the Bombardier MICAS-S2 control system.

Built during an internship at Indian Railways (ELS/BL shed), targeting depot maintenance engineers.

---

## What it does

Takes a raw DDS (Diagnostic Data System) Excel export and automatically:

- Detects operational sessions (power cycles) across the full log
- Identifies fault chains that persist or recur across multiple sessions
- Localises faults to specific physical circuit boards using ECode, Event Name, and EnvBl columns
- Ranks suspected components against a 3-year ATIL fleet failure register (132 Line+ICMS records)
- Prioritises incidents by severity and recency for shed action

**What it does that manual review can't:** cross-session chain tracking, fleet-level ATIL evidence ranking, and automatic board-level localisation — across logs of 1,000+ events spanning weeks of operation.

---

## How to use

1. Open the app URL
2. Upload a DDS Excel export (`.xlsx`) from the DDS software
3. Read the prioritised fault verdicts in the dashboard

The tool works on DDS exports from WAG-9/WAP-5/WAP-7 locos running MICAS-S2. Tested across 15+ locomotive logs.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure

```
app.py                  — Streamlit dashboard (main entry point)
dds_sessionizer.py      — Hybrid power-cycle session detection
dds_chain_matcher.py    — Fault chain pattern library and matching
dds_cross_session.py    — Cross-session chain tracking and confirmation
dds_board_localizer.py  — FFM hardware board localisation (ECode/EnvBl taxonomy)
atil_evidence_engine.py — ATIL fleet failure register evidence ranking
dds_integration_test.py — Integration test runner (not needed for deployment)
```

---

## Reference

FFM document: 3EH-214057-0001 (Bombardier MICAS-S2 Fault Finding Manual)  
ATIL register: ELS/BL shed, Line+ICMS failures, April 2024 – November 2026

---

## Limitations

- Requires DDS Excel export format — cannot read live sensor streams from the DDS software directly
- Board localisation confidence depends on ECode completeness in the export
- ATIL evidence weights are based on one shed's failure history; fleet-wide patterns may differ
