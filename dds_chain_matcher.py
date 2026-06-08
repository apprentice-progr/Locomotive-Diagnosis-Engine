"""
DDS Locomotive Fault Analysis — Causal Chain Matcher
=====================================================
Step 1 of the diagnostic pipeline.

Defines known failure chains grounded in actual DDS log observations
from IR WAG9/WAP7 fleet (BL shed), cross-referenced with Bombardier FFM.

Each chain has:
  - trigger:       first detectable event(s) — what to watch for
  - propagation:   intermediate events confirming the chain is active
  - terminal:      end-state events (isolation, shutdown, full failure)
  - max_window:    maximum minutes across which the chain can span
  - dcu_aware:     whether DCU1/DCU2 asymmetry matters for this chain
  - context_flags: extra signals that confirm or modify the chain reading

Usage:
    python dds_chain_matcher.py <path_to_dds_excel>

Output:
    Prints matched chains per session, with timing, confidence, and
    actionable notes — no visual output, plain text only.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# CHAIN DEFINITIONS
# ---------------------------------------------------------------------------
# Match strategy per event:
#   "exact"   — full Dist Text match (case-insensitive strip)
#   "contains" — substring match on Dist Text
#   "event"   — match on Event Name column
#   "ecode"   — match on ECode 0 column (hex string)
#   "envbl"   — match on EnvBl Id column
#
# A chain fires when at least one trigger matches, then looks for
# propagation and/or terminal events within max_window minutes.
# Confidence rises with each confirmed propagation/terminal step.

CHAIN_LIBRARY = [

    # ------------------------------------------------------------------
    # CHAIN 1: Coolant Pressure Degradation → FPGA Protective Shutdown
    # Observed in: IRPRP43771, IRPRP42012, IRPRP37602
    # Mechanism: coolant pressure drops (pump/sensor) → DCU thermal
    #   protection activates → FPGA watchdog kills M1/M2/M3 boards →
    #   PA2 protective shutdown fires on the originating DCU.
    # Key insight: FPGA shutdowns are CONSEQUENCES. The coolant event
    #   is the actionable root. Check which DCU the coolant event
    #   originated from (ECode prefix 3xxx=DCU1, 4xxx=DCU2).
    # ------------------------------------------------------------------
    {
        "chain_id":    "COOLANT_FPGA",
        "name":        "Coolant Pressure Drop → FPGA Protective Shutdown",
        "subsystem":   "Traction Converter (DCU Cooling)",
        "dcu_aware":   True,
        "max_window":  60,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Coolant pressure 1 below minimum"},
            {"match": "contains", "field": "Dist Text",
             "value": "Coolant pressure 1 low"},
            {"match": "contains", "field": "Dist Text",
             "value": "CON1 Coolant Pressure Below Limit"},
            {"match": "event",    "field": "Event Name",
             "value": "D_XQCo1_MinSPIF"},
            {"match": "event",    "field": "Event Name",
             "value": "D_XQCo1_LowSPIF"},
            {"match": "event",    "field": "Event Name",
             "value": "XO1_ECoolPmp1pLo"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Coolant temp. above level"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdSPIF"},        # PA2 protective shutdown
        ],
        "terminal": [
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdByFpgaM1"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdByFpgaM2"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdByFpgaM3"},
            {"match": "contains", "field": "Dist Text",
             "value": "FPGA caused PS"},
            {"match": "contains", "field": "Dist Text",
             "value": "Subsystem 02 & 03 off"},
            {"match": "event",    "field": "Event Name",
             "value": "IN_EIsoSubS02andSubS03"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Root cause is coolant circuit, NOT the FPGA shutdown. "
            "Check coolant pump MCB (63.1/1 in HB1 for DCU1, 63.1/2 in HB2 for DCU2). "
            "Check expansion tank level. Inspect oil pipeline and OCB casing. "
            "Try BUR-II isolation (MCB 127.22/2 in SB-2) to rule out false pressure reading. "
            "Do not reset FPGA faults without first confirming coolant circuit is healthy."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 2: VCB Stuck ON → Repeated Trip Attempts → Bogie Isolation
    # Observed in: IRPRP30821 (75 occurrences), IRPRP37571
    # Mechanism: VCB de-energised but aux contacts still show CLOSED →
    #   system retries → timeout → SS01 isolated → panto forced down.
    # If also seeing Timeout LC pulse: line converter is also failing
    #   to respond, suggesting a wider MCE board issue, not just VCB.
    # ------------------------------------------------------------------
    {
        "chain_id":    "VCB_STUCK_ON",
        "name":        "VCB Stuck ON → SS01 Isolation",
        "subsystem":   "Main Power / VCB",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "VCB will not open"},
            {"match": "event",    "field": "Event Name",
             "value": "MC_EMCBStkOn"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "5071"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Timeout LC pulse"},
            {"match": "event",    "field": "Event Name",
             "value": "AM1_EToPulsCvBl"},
            {"match": "event",    "field": "Event Name",
             "value": "AM2_EToPulsCvBl"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01 main power off"},
            {"match": "contains", "field": "Dist Text",
             "value": "MCE off - pan was down"},
            {"match": "event",    "field": "Event Name",
             "value": "CCUO1_54_BSubS01_Off_Diag"},
            {"match": "event",    "field": "Event Name",
             "value": "MPV_EPgDown10Min"},
        ],
        "severity":    "HIGH",
        "action":      (
            "VCB mechanically stuck or aux contact circuit faulty. "
            "Press BLDJ to attempt close. Check VCB (Pos. 5) armature and contacts physically. "
            "Check pneumatic supply to VCB trip coil. "
            "If Timeout LC pulse also present: check HBB1/STB1 processor output — "
            "may indicate MCE board issue beyond just VCB. "
            "Do NOT force-reset without physical inspection."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 3: OHE Voltage Fault → DC Link Disturbance → MVB Cascade
    # Observed in: IRPRP30821 (Feb 13 incident), IRPRP43771, IRPRP42012
    # Mechanism: OHE dip/loss → line converter loses input → DC link
    #   voltage collapses or spikes → one DCU generates MVB pullwire
    #   signal → healthy DCU also shuts down (cross-DCU cascade).
    # Key: S:0009-MVB pullwire confirms cascade. The originating DCU
    #   is the one whose ECode prefix differs (3xxx vs 4xxx).
    # PSPW w/o error cause + GPBPW w/o error cause = intermittent
    #   hardware (capacitor/connector) rather than clean software fault.
    # ------------------------------------------------------------------
    {
        "chain_id":    "OHE_MVB_CASCADE",
        "name":        "OHE Voltage Fault → DC Link → MVB Cross-DCU Cascade",
        "subsystem":   "Main Power / Line Converter",
        "dcu_aware":   True,
        "max_window":  10,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Primary voltage below minimum"},
            {"match": "contains", "field": "Dist Text",
             "value": "Catenary Voltage out of Limits"},
            {"match": "contains", "field": "Dist Text",
             "value": "AC line voltage plaus. fault"},
            {"match": "contains", "field": "Dist Text",
             "value": "Low Frequency Oscillations in line voltage"},
            {"match": "event",    "field": "Event Name",
             "value": "MNV_EULnLtLim"},
            {"match": "event",    "field": "Event Name",
             "value": "MNV_EULnOutRange"},
            {"match": "event",    "field": "Event Name",
             "value": "D_UAcLnPlyFlSPIF"},
            {"match": "event",    "field": "Event Name",
             "value": "D_UAcLoFrDt"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "DC link voltage UDC2>Umax"},
            {"match": "contains", "field": "Dist Text",
             "value": "Timeout pulse enable bogie"},
            {"match": "event",    "field": "Event Name",
             "value": "AM1_ETmOutPulsBogie"},
            {"match": "event",    "field": "Event Name",
             "value": "AM2_ETmOutPulsBogie"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "MVB pullwire from other DCU active"},
            {"match": "contains", "field": "Dist Text",
             "value": "PSPW w/o error cause"},
            {"match": "contains", "field": "Dist Text",
             "value": "GPBPW w/o error cause"},
            {"match": "event",    "field": "Event Name",
             "value": "D_MvbPwAvSpifXSPIF"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdPwAvL"},
            {"match": "event",    "field": "Event Name",
             "value": "D_FsSd_GpbPwL"},
        ],
        "severity":    "MEDIUM",
        "intermittent_flag": [
            "PSPW w/o error cause",
            "GPBPW w/o error cause",
        ],
        "action":      (
            "If OHE-caused: check OHE voltmeter, 2A fuse at SB-1. "
            "Correlate with other locos on same section for infrastructure issue. "
            "If MVB pullwire present: identify originating DCU from ECode prefix "
            "(3xxx=DCU1, 4xxx=DCU2). Inspect that DCU's SPIF board and MVB cabling. "
            "If PSPW/GPBPW w/o error cause present: suspect intermittent hardware "
            "(capacitor degradation or connector oxidation on line converter board). "
            "Do NOT treat as pure OHE issue if MVB pullwire fires consistently."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 4: Repeated VCB Cycling → Precharge Resistor Overheating
    # Observed in: IRPRP41643, IRPRP30821, IRPRP42012
    # Mechanism: driver or technician cycling MCE power repeatedly to
    #   recover from another fault → precharge resistor (limits inrush
    #   on DC link capacitors) overheats → charging disabled →
    #   CON wait precharge timeout fires.
    # This is always a SECONDARY chain — find what fault caused the
    #   repeated cycling in the first place.
    # ------------------------------------------------------------------
    {
        "chain_id":    "PRECHARGE_OVERHEAT",
        "name":        "Repeated Power Cycling → Precharge Resistor Overheat",
        "subsystem":   "Line Converter / DC Link",
        "dcu_aware":   True,
        "max_window":  120,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Charging disabled, resistor too hot"},
            {"match": "event",    "field": "Event Name",
             "value": "D_DsDcLkChL"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "CON1 wait precharge wait.time"},
            {"match": "contains", "field": "Dist Text",
             "value": "CON2 wait precharge wait.time"},
            {"match": "event",    "field": "Event Name",
             "value": "AM1_EPreCgWaitRun"},
            {"match": "event",    "field": "Event Name",
             "value": "AM2_EPreCgWaitRun"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "AC separation contactor closing flt"},
            {"match": "event",    "field": "Event Name",
             "value": "D_AcSrCtOnFlL"},
        ],
        "severity":    "MEDIUM",
        "action":      (
            "This is a secondary fault — identify what caused repeated MCE cycling. "
            "Allow 15-20 min cooldown before attempting restart. "
            "Check DC link voltage sensor (CON1-U331) if precharge keeps timing out "
            "after cooldown. If AC separation contactor fault also present: "
            "inspect contactor CON1-K101 and its drive circuit on the line converter board. "
            "Root cause is upstream — do not treat precharge overheat as primary fault."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 5: BUR Lifesign Loss → Aux Converter Isolation
    # Observed in: IRPRP37571, IRPRP42012, IRPRP43771
    # Mechanism: FLG loses MVB lifesign from BUR1/2/3 → SS06/07/08
    #   isolated → reduced ventilation and/or battery charging affected.
    # BUR2 loss specifically matters for battery charging over time.
    # All three BURs losing simultaneously = aux winding input issue.
    # ------------------------------------------------------------------
    {
        "chain_id":    "BUR_LIFESIGN_LOSS",
        "name":        "BUR Lifesign Loss → Auxiliary Converter Isolation",
        "subsystem":   "Auxiliary Converter",
        "dcu_aware":   False,
        "max_window":  15,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B1AUXC1 missing"},
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B2AUXC1 missing"},
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B3AUXC1 missing"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_EMVBDistBUR1"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_EMVBDistBUR2"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_EMVBDistBUR3"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Inverter fault"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_EInvFltBUR1"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_EInverterFltBUR2"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Isolation Demand SS06 from BUR1"},
            {"match": "contains", "field": "Dist Text",
             "value": "Isolation demand SS07 from BUR2"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS06 auxiliary converter1 off"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR1_06_BSS06IsoDem"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR2_06_BSS07IsoDem"},
            {"match": "event",    "field": "Event Name",
             "value": "CCUO1_54_BSubS06_Off"},
        ],
        "severity":    "MEDIUM",
        "all_three_flag": [
            "Lifesign from B1AUXC1 missing",
            "Lifesign from B2AUXC1 missing",
            "Lifesign from B3AUXC1 missing",
        ],
        "action":      (
            "Press BLDJ. Check MCB 127.22/1 (SB-1) for BUR1, 127.22/2 (SB-2) for BUR2, "
            "127.22/3 (SB-2) for BUR3. Reset once after MCE OFF. "
            "Check fibre optic cable FLG to affected BUR rack. "
            "If all three BURs lose lifesign simultaneously: check auxiliary winding "
            "input voltage — common-cause failure, not three independent faults. "
            "BUR2 loss specifically: monitor battery voltage — charging will degrade "
            "over time. Below 86V converters switch off. Below 82V: relief loco needed."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 6: Main Reservoir Pressure Low → Brake Interlock → No Traction
    # Observed in: IRPRP37571, IRPRP42012, IRPRP41643
    # Mechanism: compressor MCB trips or compressor fails → MR pressure
    #   drops below 5.6 kg/cm2 → brake interlock fires → no traction
    #   possible until 6.4 kg/cm2 restored.
    # If both compressor MCBs trip: usually earth fault or overload
    #   in compressor motor — do NOT keep resetting.
    # ------------------------------------------------------------------
    {
        "chain_id":    "MR_PRESSURE_BRAKE",
        "name":        "Main Reservoir Pressure Low → Brake Interlock",
        "subsystem":   "Pneumatic / Braking",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "S/R interlock - main res. low"},
            {"match": "event",    "field": "Event Name",
             "value": "XCV_EpMnResNOK"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "5029"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "S/R interlock - loco brake"},
            {"match": "contains", "field": "Dist Text",
             "value": "S/R interlock - auto brake"},
            {"match": "contains", "field": "Dist Text",
             "value": "S/R interlock - emgbrk out"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Emergency stop - shutdown"},
            {"match": "contains", "field": "Dist Text",
             "value": "Emergency brake vigilance"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Check compressor MCBs 47.1/1 (HB1) and 47.1/2 (HB2). "
            "Reset once each after opening VCB. If MCB trips again: DO NOT reset — "
            "likely earth fault or motor overload. Work with one compressor. "
            "Check auto drain valves under main reservoirs for leakage. "
            "Check air dryer condition. If MCBs are closed but compressors not running: "
            "switch electronics OFF/ON. One compressor is sufficient for operation."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 7: IGBT Feedback Failure → Bogie Isolation (Deterministic)
    # Observed in: IRPRP43771 (Sessions 65-67, fires 28-31 sec after MCE ON)
    # Mechanism: IGBT gate driver feedback circuit fault → M1 board
    #   cannot confirm IGBT state → bogie isolated as safety measure.
    # Deterministic timing (same fault within 30 sec of every power-on)
    #   = confirmed hardware fault, not transient. Component replacement needed.
    # ------------------------------------------------------------------
    {
        "chain_id":    "IGBT_FEEDBACK",
        "name":        "IGBT Feedback Failure → Motor Converter Isolation",
        "subsystem":   "Traction Converter (Motor Side)",
        "dcu_aware":   True,
        "max_window":  5,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "IGBT4 feedback failure at off order"},
            {"match": "contains", "field": "Dist Text",
             "value": "IGBT feedback failure"},
            {"match": "event",    "field": "Event Name",
             "value": "D_IGTFbFl4OfM1"},
            {"match": "event",    "field": "Event Name",
             "value": "D_IGTFbFl4OfM2"},
            {"match": "event",    "field": "Event Name",
             "value": "D_IGTFbFl4OfM3"},
        ],
        "propagation": [],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS02 traction bogie1 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS03 traction bogie2 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "Subsystem 02 & 03 off"},
            {"match": "event",    "field": "Event Name",
             "value": "CCUO1_54_BSubS02_Off"},
            {"match": "event",    "field": "Event Name",
             "value": "CCUO1_54_BSubS03_Off"},
        ],
        "severity":    "HIGH",
        "deterministic_check": True,   # flag for timing consistency check
        "action":      (
            "IGBT gate driver feedback circuit fault. Inspect IGBT driver card "
            "(Card 1669 Dual IGBT Driver) on the affected motor converter board. "
            "Check IGBT Power Module (CON1-A102 for M1) for burst condition. "
            "If fault fires within 30 seconds of EVERY power-on attempt: "
            "hardware replacement is needed — no recovery by reset. "
            "Identify which motor converter (M1/M2/M3) from ECode prefix "
            "(31xx=M1, 32xx=M2, 36xx=M3). Bogie can be isolated to limp to shed."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 8: Fire Detection System Fault (Persistent)
    # Observed in: IRPRP43771 (83 consecutive days), IRPRP42012
    # Mechanism: fire detection equipment hardware fault → persistent
    #   alarm generation → crew alarm fatigue → real fire alarm ignored.
    # This is specifically dangerous because it desensitises crew to
    #   the fire alarm signal. Must be flagged as persistent if seen
    #   across multiple sessions spanning > 7 days.
    # ------------------------------------------------------------------
    {
        "chain_id":    "FIRE_DETECT_PERSISTENT",
        "name":        "Fire Detection Equipment Fault (Persistent Alarm)",
        "subsystem":   "Fire Detection",
        "dcu_aware":   False,
        "max_window":  1440,   # 24 hours — persistence check, not propagation
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Fire alarm"},
            {"match": "contains", "field": "Dist Text",
             "value": "fire detect"},
            {"match": "event",    "field": "Event Name",
             "value": "SF_EFiAi"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "502D"},
        ],
        "propagation": [],
        "terminal": [],
        "severity":    "HIGH",
        "persistence_check": True,
        # Severity escalates to safety-critical only after persistence_critical_days.
        # A one-off fire alarm is a maintenance item.
        # 7+ days persistent = crew alarm fatigue = real safety risk.
        "persistence_critical_days": 7,
        "action":      (
            "Single/short occurrence: inspect fire detection sensor (pos. 212) and wiring. "
            "Reset alarm and monitor — if it does not recur within 24h, no further action. "
            "Persistent (7+ days): safety-critical — repeated false alarms cause crew alarm fatigue, "
            "meaning a real fire event will be missed. Replace fire detection module (pos. 212). "
            "Flag loco unfit for line duty until certified alarm-free for >24h."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 9: Transformer Temperature / Oil Circuit Fault
    # Observed in: IRPRP43771 (transformer oil circuit both), IRPRP37571
    # Mechanism: oil pump MCB trips or oil level low → transformer temp
    #   rises → TE/BE reduced → VCB trips if temp exceeds 84°C.
    # ------------------------------------------------------------------
    {
        "chain_id":    "TRAFO_OIL",
        "name":        "Transformer Oil Circuit Fault → Temperature Trip",
        "subsystem":   "Transformer / Oil Cooling",
        "dcu_aware":   False,
        "max_window":  45,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Dist. both trafo oil circuits"},
            {"match": "contains", "field": "Dist Text",
             "value": "Dist. one trafo oil circuit"},
            {"match": "contains", "field": "Dist Text",
             "value": "Disturb.Temp.sensor"},
            {"match": "event",    "field": "Event Name",
             "value": "MT_EBothTfOilPmpFlr"},
            {"match": "event",    "field": "Event Name",
             "value": "MT_ETf1NotValid"},
            {"match": "event",    "field": "Event Name",
             "value": "MT_ETf2NotValid"},
            {"match": "event",    "field": "Event Name",
             "value": "MT_ETf3NotValid"},
            {"match": "event",    "field": "Event Name",
             "value": "MT_ETf4NotValid"},
        ],
        "propagation": [],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01 main power off"},
            {"match": "event",    "field": "Event Name",
             "value": "CCUO1_54_BSubS01_Off_Diag"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Check transformer oil pump MCBs 62.1/1 (HB1) and 62.1/2 (HB2). "
            "Reset once after opening VCB. Check oil level in expansion tanks. "
            "Inspect OCB casing for damage. Check MCBs 59.1/1, 59.1/2 (oil cooler blowers). "
            "If Disturb.Temp.sensor (not actual temp fault): likely sensor circuit fault "
            "— check SLG temp sensor wiring. Both circuits failing simultaneously: "
            "check BUR output voltage balance as common cause."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 10: Panto Pressure / MCE Off Sequence
    # Observed in: IRPRP37571, IRPRP43771, IRPRP42012
    # Mechanism: pan pressure lost or VCB trip forces pan down →
    #   10-minute timer starts → MCE switches off.
    # Often a CONSEQUENCE of other chains (OHE voltage, coolant, VCB).
    # ------------------------------------------------------------------
    {
        "chain_id":    "PANTO_MCE_OFF",
        "name":        "Pantograph Pressure Loss → MCE Auto Shutdown",
        "subsystem":   "Pantograph / Main Power",
        "dcu_aware":   False,
        "max_window":  20,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "No pressure in pan"},
            {"match": "event",    "field": "Event Name",
             "value": "MPV_EPg1NoPres"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "508B"},
        ],
        "propagation": [],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "MCE off - pan was down 10 min"},
            {"match": "event",    "field": "Event Name",
             "value": "MPV_EPgDown10Min"},
        ],
        "severity":    "MEDIUM",
        "action":      (
            "Check aux reservoir pressure (>5.2 kg/cm2). "
            "Check MCB 48.1 in SB-2. Verify IG-38 blue key horizontal. "
            "Tap pressure switch No. 26 on pneumatic panel. "
            "Try alternate pantograph. "
            "If this chain follows another chain (VCB stuck, OHE fault): "
            "it is a consequence, not a root cause — resolve the upstream chain first."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 11: Card 2000-138 / CAN Communication Failure → BUR Isolation
    # ATIL grounding: 21 occurrences across fleet, most common card failure
    #   after Card 1669. Appears in BUR-1/2/3 aux converters.
    # DDS precursor from ATIL investigation notes: "CAN communication error"
    #   precedes BUR isolation. Card 2000-138 is the SCR/Converter Control
    #   Digital Controller — it handles the CAN bus interface in the BUR.
    # Guidance: rack-level inspection first (BUR-1/2/3 card cage), not
    #   immediate replacement — CAN errors can also be fibre optic/connector.
    # ------------------------------------------------------------------
    {
        "chain_id":    "CAN_COMM_BUR",
        "name":        "CAN Communication Fault → BUR Control Loss",
        "subsystem":   "Auxiliary Converter (BUR Control)",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CAN communication error"},
            {"match": "contains", "field": "Dist Text",
             "value": "CAN comm"},
            {"match": "event",    "field": "Event Name",
             "value": "CAN_ECommErr"},
            {"match": "event",    "field": "Event Name",
             "value": "XU_ECanCommErr"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B"},
            {"match": "contains", "field": "Dist Text",
             "value": "Inverter fault"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS06 auxiliary converter1 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS07 auxiliary converter2 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS08 auxiliary converter3 off"},
        ],
        "severity":    "HIGH",
        "action":      (
            "ATIL fleet history: most likely Card 2000-138 (SCR/Converter Control) in affected BUR. "
            "Before replacing card: check fibre optic cable to BUR rack and connector seating — "
            "CAN errors can be cable faults. "
            "If cable checks clean: inspect BUR card cage, check Card 2000-138 visually. "
            "Replace Card 2000-138 only if visual inspection shows damage or cable checks are clear. "
            "BUR rack location: BUR-1 in SB-1, BUR-2/3 in SB-2."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 12: DC Link Overvoltage → Card 1703 / Thyristor Driver Fault
    # ATIL grounding: 10 occurrences, second most common single-card failure.
    #   Card 1703 is the Single Thyristor Driver in BUR converters.
    # DDS precursor: "DC Link over voltage" or "DC link shoot up" in AC context.
    # Note: "half cycle firing" in investigation notes = thyristor asymmetry
    #   caused by degraded Card 1703 gate drive circuit.
    # ------------------------------------------------------------------
    {
        "chain_id":    "DC_LINK_OV_THYRISTOR",
        "name":        "DC Link Overvoltage → Thyristor Driver Fault (BUR)",
        "subsystem":   "Auxiliary Converter (BUR Thyristor)",
        "dcu_aware":   False,
        "max_window":  20,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "DC Link over voltage"},
            {"match": "contains", "field": "Dist Text",
             "value": "DC link overvoltage"},
            {"match": "contains", "field": "Dist Text",
             "value": "DC Link shoot"},
            {"match": "contains", "field": "Dist Text",
             "value": "VBph High"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR1_EUdcOv"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR2_EUdcOv"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR3_EUdcOv"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "half cycle"},
            {"match": "contains", "field": "Dist Text",
             "value": "output voltage"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS06 auxiliary converter1 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS07 auxiliary converter2 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS08 auxiliary converter3 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "auxiliary converter"},
        ],
        "severity":    "HIGH",
        "action":      (
            "ATIL fleet history: most likely Card 1703 (Single Thyristor Driver) in affected BUR. "
            "Check which BUR is isolated (B1/B2/B3 from Dist Text prefix). "
            "Inspect Card 1703 in that BUR's card cage. "
            "A single DC link overvoltage without recurrence: may be OHE transient — monitor. "
            "If fault recurs after reset or 'half cycle firing' appears: Card 1703 replacement needed. "
            "Do not replace both cards simultaneously — replace one, test, then decide on second."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 13: VCESAT Protection → IGBT Module / Card 1669 Failure
    # ATIL grounding: mentioned in investigation notes for both Line and ICMS
    #   failures. VCESAT (Collector-Emitter Saturation Voltage) protection
    #   fires when an IGBT is degraded — collector voltage stays high during
    #   conduction, indicating the device is not fully switching.
    # Distinct from IGBT_FEEDBACK (timing-based): VCESAT fires mid-operation,
    #   not at startup. Both point to the same Card 1669 gate driver.
    # ------------------------------------------------------------------
    {
        "chain_id":    "VCESAT_IGBT",
        "name":        "VCESAT Protection Trip → IGBT Degradation",
        "subsystem":   "Traction Converter (Motor Side)",
        "dcu_aware":   True,
        "max_window":  15,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "VCESAT protection"},
            {"match": "contains", "field": "Dist Text",
             "value": "Vce-Lock"},
            {"match": "contains", "field": "Dist Text",
             "value": "VCE sat"},
            {"match": "event",    "field": "Event Name",
             "value": "D_VceSatM1"},
            {"match": "event",    "field": "Event Name",
             "value": "D_VceSatM2"},
            {"match": "event",    "field": "Event Name",
             "value": "D_VceSatM3"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "IGBT feedback"},
            {"match": "contains", "field": "Dist Text",
             "value": "Current limitation"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS02 traction bogie1 off"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS03 traction bogie2 off"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdByFpgaM1"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PrSdByFpgaM2"},
        ],
        "severity":    "HIGH",
        "action":      (
            "VCESAT protection = IGBT not fully switching. This indicates IGBT degradation, "
            "not a transient. "
            "Identify which motor converter from ECode prefix (31xx=M1, 32xx=M2, 36xx=M3 on DCU1; "
            "41xx/42xx/46xx on DCU2). "
            "Inspect Card 1669 (Dual IGBT Driver) in that converter's card cage — "
            "check for burn marks, discolouration, or deformed components. "
            "If Card 1669 appears intact, the IGBT Power Module itself may be degraded. "
            "Do not attempt to reset and run — VCESAT repeated = hardware replacement needed. "
            "Bogie can be isolated to limp to shed (ATIL confirmed recovery method)."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 14: Card 1302-1 Fault → BUR Lifesign Loss + Coolant Pressure
    # ATIL grounding: Card 1302-1 (CHBA Interface / BUR Control Interface)
    #   appears in multiple records alongside "Lifesign Missing from BUR" +
    #   "coolant pressure low" as a combined presentation.
    # This is a distinct pattern from the standard BUR_LIFESIGN_LOSS chain:
    #   the coolant pressure fault appearing alongside BUR lifesign loss
    #   points to a common-cause interface card failure, not two independent
    #   faults on separate subsystems.
    # ------------------------------------------------------------------
    {
        "chain_id":    "BUR_COOLANT_INTERFACE",
        "name":        "BUR Lifesign + Coolant Pressure → Interface Card Fault",
        "subsystem":   "Auxiliary Converter / Cooling Interface",
        "dcu_aware":   False,
        "max_window":  15,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B2AUXC1 missing"},
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B1AUXC1 missing"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Coolant pressure"},
            {"match": "contains", "field": "Dist Text",
             "value": "coolent pressure"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS06 auxiliary converter"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS07 auxiliary converter"},
        ],
        "severity":    "HIGH",
        "action":      (
            "BUR lifesign loss WITH simultaneous coolant pressure fault = "
            "interface card failure, not two independent faults. "
            "ATIL fleet history points to Card 1302-1 (CHBA Interface Card) in the BUR rack. "
            "Check Card 1302-1 in SB-1 (BUR-1) or SB-2 (BUR-2/3) before investigating "
            "coolant circuit separately. "
            "If only BUR lifesign fires without coolant fault: use BUR_LIFESIGN_LOSS chain action instead."
        ),
    },

]


# ---------------------------------------------------------------------------
# SESSION CLASSIFIER
# ---------------------------------------------------------------------------
# Before chain matching, classify each MCE session so the chain matcher
# knows what context it is operating in.

SESSION_TYPES = {
    "FAULT_INVESTIGATION": "Active fault investigation — same fault fires repeatedly after MCE ON",
    "ACTIVE_TEST":         "Technician test — deliberate isolation sequences, short sessions",
    "OPERATIONAL":         "Operational running — OHE events, brake interlocks, movement context",
    "PARKED_IDLE":         "Parked/idle — energy saving, OHE persistence, no MCE cycling",
}

TEST_INDICATORS = [
    "Shunting mode", "Loco in shunting mode",
    "ZTEL operated", "Simulation",
    "Loco in banking operation", "LOCO IS IN BANKING MODE",
    "Rotary switch bogie cut out",
]

OPERATIONAL_INDICATORS = [
    "S/R interlock - brake cock",
    "S/R interlock - loco brake",
    "S/R interlock - auto brake",
    "S/R interlock - emgbrk out",
    "Primary voltage below minimum",
    "Catenary Voltage out of Limits",
    "Emergency brake vigilance",
    "Vigilance Cut Off",
    "ACP/Train Part",
    "Alarm chain pulling",
    "Overspeed",
]


def classify_session(session_df, session_duration_min):
    """
    Classify a session (MCE ON → next MCE ON) into one of the SESSION_TYPES.
    Returns (session_type_key, reason_string).
    """
    texts = session_df["Dist Text"].fillna("").str.lower()
    n_events = len(session_df)
    p1_count = (session_df["Prio"] == 1).sum()

    # Parked: long, sparse, no P1 action
    if session_duration_min > 300 and n_events < 20 and p1_count < 3:
        return "PARKED_IDLE", f"Duration {session_duration_min:.0f}min, only {n_events} events"

    # Active test: deliberate isolation language present
    for kw in TEST_INDICATORS:
        if any(kw.lower() in t for t in texts):
            return "ACTIVE_TEST", f"Test indicator found: '{kw}'"

    # Fault investigation: look for same P1 fault firing multiple times
    # in a short session — characteristic of repeated power-on attempts
    if session_duration_min < 60 and p1_count >= 2:
        p1_texts = session_df[session_df["Prio"] == 1]["Dist Text"].value_counts()
        if len(p1_texts) > 0 and p1_texts.iloc[0] >= 2:
            return "FAULT_INVESTIGATION", (
                f"Fault '{p1_texts.index[0]}' fires {p1_texts.iloc[0]}x "
                f"in {session_duration_min:.0f}min session"
            )

    # Operational: brake interlocks, OHE events, vigilance
    for kw in OPERATIONAL_INDICATORS:
        if any(kw.lower() in t for t in texts):
            return "OPERATIONAL", f"Operational indicator found: '{kw}'"

    return "FAULT_INVESTIGATION", "Default: P1 faults present, not otherwise classified"


# ---------------------------------------------------------------------------
# CHAIN MATCHER
# ---------------------------------------------------------------------------

def _event_matches(row, condition):
    """Test whether a single DataFrame row matches one condition dict."""
    field = condition["field"]
    value = condition["value"].lower()
    match = condition["match"]
    cell  = str(row.get(field, "")).lower().strip()

    if match == "exact":
        return cell == value
    if match == "contains":
        return value in cell
    if match == "event":
        # Prefix match — Event Name often has suffix like _CON12
        return cell.startswith(value.lower())
    if match == "ecode":
        return cell == value.lower()
    if match == "envbl":
        return cell == value.lower()
    return False


def _matches_any(row, conditions):
    """Return True if row matches at least one condition in the list."""
    return any(_event_matches(row, c) for c in conditions)


def _find_first_match(df, conditions):
    """Return (index, row) of first matching event or (None, None)."""
    for idx, row in df.iterrows():
        if _matches_any(row, conditions):
            return idx, row
    return None, None


def match_chains(session_df, chain_library):
    """
    Attempt to match all chains against a session DataFrame.

    Returns a list of match dicts with timing and confidence.
    """
    matches = []
    session_df = session_df.sort_values("Start time").reset_index(drop=True)

    for chain in chain_library:
        # Skip chains with no trigger conditions
        if not chain.get("trigger"):
            continue

        # Find trigger event
        trig_idx, trig_row = _find_first_match(session_df, chain["trigger"])
        if trig_idx is None:
            continue

        t_trigger = trig_row["Start time"]
        window    = pd.Timedelta(minutes=chain["max_window"])
        window_df = session_df[
            (session_df["Start time"] >= t_trigger) &
            (session_df["Start time"] <= t_trigger + window)
        ]

        # Check propagation
        prop_hits  = []
        for cond in chain.get("propagation", []):
            idx, row = _find_first_match(window_df, [cond])
            if idx is not None:
                prop_hits.append({
                    "text": row["Dist Text"],
                    "time": row["Start time"],
                    "lag_min": round(
                        (row["Start time"] - t_trigger).total_seconds() / 60, 1
                    ),
                })

        # Check terminal
        term_hits = []
        for cond in chain.get("terminal", []):
            idx, row = _find_first_match(window_df, [cond])
            if idx is not None:
                term_hits.append({
                    "text": row["Dist Text"],
                    "time": row["Start time"],
                    "lag_min": round(
                        (row["Start time"] - t_trigger).total_seconds() / 60, 1
                    ),
                })

        # Confidence scoring
        # Trigger alone = 0.3, each prop step = +0.2, each terminal = +0.25
        # Cap at 1.0
        confidence = 0.3
        confidence += min(len(prop_hits) * 0.2, 0.4)
        confidence += min(len(term_hits) * 0.25, 0.5)
        confidence = round(min(confidence, 1.0), 2)

        # DCU origination
        dcu_origin = None
        if chain.get("dcu_aware"):
            ecode = str(trig_row.get("ECode 0", "")).strip()
            if ecode.startswith("3"):
                dcu_origin = "DCU1"
            elif ecode.startswith("4"):
                dcu_origin = "DCU2"

        # Deterministic check: same fault fires repeatedly within session
        is_deterministic = False
        if chain.get("deterministic_check"):
            repeat_df = session_df[
                session_df["Dist Text"] == trig_row["Dist Text"]
            ]
            if len(repeat_df) >= 2:
                lags = repeat_df["Start time"].diff().dt.total_seconds().dropna()
                if lags.std() < 30:  # < 30 sec std dev = deterministic
                    is_deterministic = True

        # Intermittent hardware flag
        is_intermittent = False
        for kw in chain.get("intermittent_flag", []):
            if session_df["Dist Text"].str.contains(kw, na=False).any():
                is_intermittent = True
                break

        # Persistence check (fire detection etc.)
        is_persistent = False
        persistence_days = 0
        if chain.get("persistence_check"):
            is_persistent = True   # the caller (process_file) will compute days

        matches.append({
            "chain_id":        chain["chain_id"],
            "name":            chain["name"],
            "subsystem":       chain["subsystem"],
            "severity":        chain["severity"],
            "confidence":      confidence,
            "trigger_text":    trig_row["Dist Text"],
            "trigger_time":    t_trigger,
            "propagation_hits": prop_hits,
            "terminal_hits":   term_hits,
            "dcu_origin":      dcu_origin,
            "is_deterministic": is_deterministic,
            "is_intermittent": is_intermittent,
            "is_persistent":   is_persistent,
            "action":          chain["action"],
        })

    # Sort by severity then confidence
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    matches.sort(
        key=lambda x: (sev_order.get(x["severity"], 9), -x["confidence"])
    )
    return matches


# ---------------------------------------------------------------------------
# PERSISTENCE CHECKER
# ---------------------------------------------------------------------------

def check_persistence(full_df, chain):
    """
    For chains with persistence_check=True, compute how many distinct
    calendar days the trigger event appears across the full log.
    Returns (n_days, first_date, last_date).
    """
    trigger_conditions = chain.get("trigger", [])
    hits = full_df[
        full_df.apply(lambda r: _matches_any(r, trigger_conditions), axis=1)
    ]
    if hits.empty:
        return 0, None, None
    dates = hits["Start time"].dt.date
    return dates.nunique(), dates.min(), dates.max()


# ---------------------------------------------------------------------------
# DETERMINISTIC TIMING ANALYSER
# ---------------------------------------------------------------------------

def analyse_deterministic_timing(full_df, fault_text, mce_on_times):
    """
    For IGBT-style faults: compute the lag between MCE ON and each
    fault firing. Returns list of lags in seconds.
    """
    lags = []
    fault_times = full_df[
        full_df["Dist Text"].str.contains(fault_text, na=False, case=False)
    ]["Start time"].tolist()

    for ft in fault_times:
        # Find the most recent MCE ON before this fault
        prior = [t for t in mce_on_times if t < ft]
        if prior:
            lag = (ft - max(prior)).total_seconds()
            if lag < 300:   # only count if < 5 min after power on
                lags.append(round(lag, 1))
    return lags


# ---------------------------------------------------------------------------
# MAIN FILE PROCESSOR
# ---------------------------------------------------------------------------

def process_file(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n{'='*70}")
    print(f"  DDS Chain Matcher — {path.name}")
    print(f"{'='*70}")

    df = pd.read_excel(file_path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
    df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

    vehicle = df["Vehicle Name"].iloc[0] if "Vehicle Name" in df.columns else path.stem
    date_min = df["Start time"].min().strftime("%d %b %Y")
    date_max = df["Start time"].max().strftime("%d %b %Y")
    span_days = (df["Start time"].max() - df["Start time"].min()).days

    print(f"  Vehicle : {vehicle}")
    print(f"  Period  : {date_min} to {date_max}  ({span_days} days)")
    print(f"  Events  : {len(df)}  |  P1: {(df['Prio']==1).sum()}  |  P2: {(df['Prio']==2).sum()}")

    # MCE session boundaries
    mce_on_df   = df[df["Dist Text"].str.contains("Power on of MCE", na=False)]
    mce_on_times = mce_on_df["Start time"].tolist()

    if not mce_on_times:
        # Fallback: treat whole file as one session
        mce_on_times = [df["Start time"].min()]

    print(f"  MCE sessions: {len(mce_on_times)}")

    # Persistence check for fire detection across whole file
    fire_chain = next((c for c in CHAIN_LIBRARY if c["chain_id"] == "FIRE_DETECT_PERSISTENT"), None)
    fire_days, fire_first, fire_last = (0, None, None)
    if fire_chain:
        fire_days, fire_first, fire_last = check_persistence(df, fire_chain)

    # Deterministic timing for IGBT faults
    igbt_lags = analyse_deterministic_timing(df, "IGBT", mce_on_times)

    # Per-session analysis
    all_chain_hits = defaultdict(list)   # chain_id -> list of session match dicts
    session_summaries = []

    for i, t_start in enumerate(mce_on_times):
        t_end = mce_on_times[i + 1] if i + 1 < len(mce_on_times) else df["Start time"].max()
        session_df = df[
            (df["Start time"] >= t_start) & (df["Start time"] < t_end)
        ].copy()

        if session_df.empty:
            continue

        duration_min = (t_end - t_start).total_seconds() / 60
        stype, sreason = classify_session(session_df, duration_min)

        chain_matches = match_chains(session_df, CHAIN_LIBRARY)

        if chain_matches or stype in ("FAULT_INVESTIGATION", "OPERATIONAL"):
            session_summaries.append({
                "session_n":  i + 1,
                "start":      t_start,
                "duration":   duration_min,
                "stype":      stype,
                "sreason":    sreason,
                "matches":    chain_matches,
                "n_events":   len(session_df),
                "p1_count":   int((session_df["Prio"] == 1).sum()),
            })

        for m in chain_matches:
            all_chain_hits[m["chain_id"]].append(m)

    # ---------------------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------------------

    print(f"\n{'─'*70}")
    print("  CHAIN SUMMARY (across all sessions)")
    print(f"{'─'*70}")

    if not all_chain_hits:
        print("  No chains matched in this log.")
    else:
        for chain_id, hits in sorted(
            all_chain_hits.items(),
            key=lambda x: (
                {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x[1][0]["severity"], 3),
                -len(x[1])
            )
        ):
            chain_def = next((c for c in CHAIN_LIBRARY if c["chain_id"] == chain_id), {})
            avg_conf  = round(sum(h["confidence"] for h in hits) / len(hits), 2)
            first_hit = min(h["trigger_time"] for h in hits)
            last_hit  = max(h["trigger_time"] for h in hits)
            dcu_origins = set(h["dcu_origin"] for h in hits if h["dcu_origin"])

            print(f"\n  [{hits[0]['severity']}] {hits[0]['name']}")
            print(f"  Chain ID   : {chain_id}")
            print(f"  Subsystem  : {hits[0]['subsystem']}")
            print(f"  Occurrences: {len(hits)}  |  Avg confidence: {avg_conf}")
            print(f"  First seen : {first_hit.strftime('%d %b %Y %H:%M')}")
            print(f"  Last seen  : {last_hit.strftime('%d %b %Y %H:%M')}")
            if dcu_origins:
                print(f"  DCU origin : {', '.join(sorted(dcu_origins))}")

            # Special flags
            if chain_id == "FIRE_DETECT_PERSISTENT" and fire_days > 0:
                print(f"  *** PERSISTENCE: Fire alarm present on {fire_days} distinct days")
                print(f"      First: {fire_first}  Last: {fire_last}")
                if fire_days >= 7:
                    print(f"      *** WARNING: {fire_days} days persistent — "
                          f"crew alarm fatigue risk. Unfit for line duty.")

            if chain_id == "IGBT_FEEDBACK" and igbt_lags:
                mean_lag = round(sum(igbt_lags) / len(igbt_lags), 1)
                std_lag  = round(float(np.std(igbt_lags)), 1)
                print(f"  *** DETERMINISTIC TIMING: {len(igbt_lags)} occurrences, "
                      f"mean {mean_lag}s after MCE ON (std={std_lag}s)")
                if std_lag < 30:
                    print(f"      *** Consistent timing confirms hardware fault, "
                          f"NOT transient. Component replacement required.")

            # Propagation and terminal evidence from best hit
            best = max(hits, key=lambda h: h["confidence"])
            if best["propagation_hits"]:
                print(f"  Propagation confirmed:")
                for p in best["propagation_hits"]:
                    print(f"    +{p['lag_min']}min  {p['text']}")
            if best["terminal_hits"]:
                print(f"  Terminal events confirmed:")
                for t in best["terminal_hits"][:3]:
                    print(f"    +{t['lag_min']}min  {t['text']}")
            if best["is_intermittent"]:
                print(f"  *** INTERMITTENT HARDWARE FLAG: "
                      f"PSPW/GPBPW w/o error cause present — "
                      f"check capacitor/connector condition")

            print(f"\n  Action: {chain_def.get('action', 'See FFM for guidance.')}")

    # Session details for flagged sessions
    flagged = [s for s in session_summaries if s["matches"]]
    if flagged:
        print(f"\n{'─'*70}")
        print(f"  FLAGGED SESSIONS ({len(flagged)} with chain matches)")
        print(f"{'─'*70}")
        for s in flagged[:20]:   # cap at 20 to keep output readable
            print(f"\n  Session {s['session_n']:3d}  |  "
                  f"{s['start'].strftime('%d %b %H:%M')}  |  "
                  f"{s['duration']:.0f}min  |  "
                  f"{s['stype']}  |  events={s['n_events']} P1={s['p1_count']}")
            print(f"  Context: {s['sreason']}")
            for m in s["matches"]:
                print(f"    → [{m['severity']}] {m['name']}  "
                      f"(conf={m['confidence']})  "
                      f"trigger: {m['trigger_text'][:70]}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python dds_chain_matcher.py <dds_excel_file> [file2] ...")
        print("\nOr run the demo against the available test files:")
        demo_files = [
            "/mnt/user-data/uploads/30821.xlsx",
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260309_044_A_3771.xlsx",
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260214_046_A_7571.xlsx",
            "/mnt/user-data/uploads/ED_V_IR_PRP___20260403_044_A_2012.xlsx",
        ]
        for f in demo_files:
            process_file(f)
    else:
        for f in sys.argv[1:]:
            process_file(f)