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

    # ------------------------------------------------------------------
    # CHAIN 19: Fuse 415/110V Circuit Blown
    # M2HR frequency: 49 occurrences across fleet — most common unmatched
    # fault type in the entire register.
    # ECode 507E, EnvBl EG_STB1_HBB1, Event MV_ECrBk415_110
    # Mechanism: the 415V/110V auxiliary supply fuse (in STB1/HBB1 cubicle)
    #   blows → auxiliary loads on that circuit lose supply → CCUO logs
    #   the loss. Physical fuse replacement required — not a card fault.
    # ------------------------------------------------------------------
    {
        "chain_id":    "FUSE_415_110V",
        "name":        "Fuse 415/110V Circuit Blown → Auxiliary Supply Loss",
        "subsystem":   "Auxiliary Supply (STB1/HBB1)",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0127"},
            {"match": "contains", "field": "Dist Text",
             "value": "Fuse 415/110V circuit blown"},
            {"match": "event",    "field": "Event Name",
             "value": "MV_ECrBk415_110"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "507E"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "auxiliary converter"},
            {"match": "contains", "field": "Dist Text",
             "value": "blower MCB"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS06"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS07"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Fuse 415V/110V circuit blown in STB1/HBB1 cubicle. "
            "Locate fuse F1 or F2 in HBB1 (Cab 1 side auxiliary supply). "
            "Replace blown fuse — check rating before replacing (do not uprate). "
            "If fuse blows again immediately after replacement: earth fault or short circuit "
            "on the 415V/110V circuit — DO NOT replace again. "
            "Isolate the circuit, identify the shorted load (check auxiliary motor windings "
            "and wiring in HB-1 cubicle), clear fault before refitting fuse. "
            "Single blow with no recurrence: may be a transient — replace and monitor."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 20: Earth Fault Control Circuit
    # M2HR frequency: 30 occurrences — safety-critical, must not be deferred.
    # ECode 507C, EnvBl EG_STB1_HBB1, Event MV_EGndFlrCtr
    # Mechanism: earth fault detected on the control circuit wiring →
    #   CCUO isolates affected circuit to prevent further damage.
    #   Can indicate wiring insulation breakdown, moisture ingress,
    #   or a failed component with its output shorted to earth.
    # ------------------------------------------------------------------
    {
        "chain_id":    "EARTH_FAULT_CTRL",
        "name":        "Earth Fault → Control Circuit Isolation",
        "subsystem":   "Control Circuit Wiring (STB1/HBB1)",
        "dcu_aware":   False,
        "max_window":  60,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0125"},
            {"match": "contains", "field": "Dist Text",
             "value": "Earth fault control circuit"},
            {"match": "event",    "field": "Event Name",
             "value": "MV_EGndFlrCtr"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "507C"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
            {"match": "contains", "field": "Dist Text",
             "value": "Subsystem"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Earth fault on control circuit — DO NOT defer. "
            "Measure insulation resistance (IR) on the control circuit wiring in STB1/HBB1 "
            "with a megger. IR < 1 MΩ confirms an earth fault. "
            "Check: (1) Wiring harness condition in HBB1 — look for chafing, moisture, "
            "or burnt insulation. "
            "(2) Control circuit contactors and relays for shorted coils. "
            "(3) Any recently replaced components in the control circuit path. "
            "If fault is intermittent (clears on reset): check connectors for moisture "
            "or corrosion — clean and reseat. "
            "Loco must not be returned to service until IR reading is confirmed healthy. "
            "Repeated earth fault after clearance: suspect wiring harness replacement needed."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 21: Compressor MCB Open (Standalone)
    # M2HR frequency: 74 occurrences combined (0142 x29, 0134 x45)
    # These fire without the MR pressure interlock — the compressor MCB
    #   trips before pressure falls enough to trigger MR_PRESSURE_BRAKE.
    # ECode 508D (comp 2, HBB2), 5085 (comp 1, HBB1)
    # Important: if BOTH compressor MCBs open (0134 variant), it's more
    #   serious — earth fault or supply issue affecting both circuits.
    # ------------------------------------------------------------------
    {
        "chain_id":    "COMPRESSOR_MCB",
        "name":        "Compressor MCB Open → Air Supply Degradation",
        "subsystem":   "Auxiliary Supply (Compressor Circuit)",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0142"},
            {"match": "contains", "field": "Dist Text",
             "value": "MCB of compressor 2 open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0134"},
            {"match": "contains", "field": "Dist Text",
             "value": "MCB of compressor 1 open"},
            {"match": "contains", "field": "Dist Text",
             "value": "MCB of compressor 1 and 2 open"},
            {"match": "event",    "field": "Event Name",
             "value": "XCV_ECrBkCmp1Off"},
            {"match": "event",    "field": "Event Name",
             "value": "XCV_ECrBkCmp2Off"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "508D"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "5085"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "main res. low"},
            {"match": "contains", "field": "Dist Text",
             "value": "S/R interlock"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "brake interlock"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0042"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Compressor MCB open. "
            "MCB 47.1/1 (HB1) = Compressor 1. MCB 47.1/2 (HB2) = Compressor 2. "
            "Reset once after opening VCB. If MCB holds: monitor MR pressure recovery "
            "(should reach 8–9 kg/cm² within 5–7 minutes). "
            "If MCB trips again immediately: DO NOT reset — likely compressor motor "
            "overload or earth fault. Work on remaining compressor. "
            "If BOTH MCBs open simultaneously: common-cause fault — check "
            "auxiliary supply to both compressor circuits before resetting either. "
            "Check auto drain valve condition and auto drain valve timer setting — "
            "excessive drain cycling can overload the compressor motor. "
            "Persistent MCB trips after reset: compressor motor winding check required."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 22: Transformer Oil Pump MCB Open (Standalone)
    # M2HR frequency: 17 occurrences (0152 x11, 0122 x6)
    # These fire without the TRAFO_OIL chain trigger — the MCB trips
    #   before temperature rises enough to trigger the temperature chain.
    # ECode 5097 (pump 2, HBB2), 5079 (pump 1, HBB1)
    # ------------------------------------------------------------------
    {
        "chain_id":    "TRAFO_PUMP_MCB",
        "name":        "Transformer Oil Pump MCB Open → Cooling Degradation",
        "subsystem":   "Transformer Cooling Circuit",
        "dcu_aware":   False,
        "max_window":  45,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0152"},
            {"match": "contains", "field": "Dist Text",
             "value": "Transformer pump 2 MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0122"},
            {"match": "contains", "field": "Dist Text",
             "value": "Transformer pump 1 MCB open"},
            {"match": "event",    "field": "Event Name",
             "value": "MT2_ECrBkOilPmp2"},
            {"match": "event",    "field": "Event Name",
             "value": "MT1_ECrBkOilPmp1"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "5097"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "5079"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Dist. one trafo oil circuit"},
            {"match": "contains", "field": "Dist Text",
             "value": "trafo oil"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Dist. both trafo oil circuits"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0082"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Transformer oil pump MCB open. "
            "MCB 62.1/1 (HB1) = Oil pump 1. MCB 62.1/2 (HB2) = Oil pump 2. "
            "Reset MCB once after opening VCB. If MCB holds and no temperature alarm "
            "follows: monitor and continue. "
            "If MCB trips again: oil pump motor fault — check pump motor winding "
            "resistance and mechanical freedom (seized bearing causes overload). "
            "Check oil level in expansion tanks — low oil level increases pump load. "
            "If BOTH pump MCBs open: check BUR output voltage balance — unbalanced "
            "BUR voltage is a common cause of simultaneous pump MCB trips. "
            "Oil cooler blower MCBs 59.1/1 and 59.1/2 should also be checked — "
            "blower fault can cause temperature rise that leads to further trips."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 23: DCU AMP Parameter Change Error
    # M2HR frequency: 15 occurrences — typically occurs after card
    #   replacement if the replacement card has a different firmware
    #   version or parameter set than the loco spec.
    # ECode 514E (DCU1), EnvBl EG_VCI, Event AM_E_ParChgAmpDCU1
    # Also appears as CCUO:0247 for DCU2.
    # ------------------------------------------------------------------
    {
        "chain_id":    "DCU_PARAM_ERROR",
        "name":        "DCU AMP Parameter Change Error → Configuration Mismatch",
        "subsystem":   "Traction Converter (VCI / Parameter Store)",
        "dcu_aware":   True,
        "max_window":  20,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0246"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0247"},
            {"match": "contains", "field": "Dist Text",
             "value": "AMP Parameter Change Error"},
            {"match": "contains", "field": "Dist Text",
             "value": "Parameter Change Error"},
            {"match": "event",    "field": "Event Name",
             "value": "AM_E_ParChgAmpDCU1"},
            {"match": "event",    "field": "Event Name",
             "value": "AM_E_ParChgAmpDCU2"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "514E"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "514F"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Iso Request CON"},
            {"match": "contains", "field": "Dist Text",
             "value": "Subsystem"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS01"},
        ],
        "severity":    "MEDIUM",
        "action":      (
            "DCU AMP parameter change error — configuration mismatch detected. "
            "Check maintenance history: has any card been recently replaced in "
            "the traction converter (CON1 or CON2)? "
            "If yes: verify that the replacement card's firmware version matches "
            "the loco's parameter spec (check loco card from TC/shed records). "
            "A card with wrong firmware will trigger this error on every MCE power-on. "
            "If no recent card replacement: check VCI board parameter memory — "
            "power surge or clock battery failure can corrupt stored parameters. "
            "Resolution: parameter re-download via DDS software, or replacement of "
            "the card with a correctly parameterised unit. "
            "Do not reset repeatedly without investigating — "
            "parameter mismatch can cause unexpected traction behaviour."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 24: BUR Output Frequency / Inverter Fault (Internal Hardware)
    # Confirmed on: 33586 (BUR3:0021 x2), 33792 (BUR3:0021 x3, BUR2:0002 x3)
    # ECode 2014 (No Output Frequency), ECode 1501 (BUR Inverter Fault)
    # EnvBl: EG_BUR3, EG_BUR2, EG_BUR1
    # DISTINCT from BUR_LIFESIGN_LOSS:
    #   - Lifesign loss = FLG can't see BUR on MVB → communication/fibre fault
    #   - Output fault  = BUR is running but output is wrong/absent → internal
    #     BUR hardware fault. Different card, different action.
    # BUR3:0021 = no output frequency from BUR3 (charge converter section).
    # BUR2:0002 = inverter fault inside BUR2 (drive electronics failed).
    # ------------------------------------------------------------------
    {
        "chain_id":    "BUR_OUTPUT_FAULT",
        "name":        "BUR Output / Inverter Fault (Internal Hardware)",
        "subsystem":   "Auxiliary Converter (BUR Internal)",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "No Output Frequency"},
            {"match": "contains", "field": "Dist Text",
             "value": "BUR3:0021"},
            {"match": "contains", "field": "Dist Text",
             "value": "BUR2:0002"},
            {"match": "contains", "field": "Dist Text",
             "value": "BUR1:0002"},
            {"match": "contains", "field": "Dist Text",
             "value": "BUR3:0002"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "2014"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "1501"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "1502"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR3_ENoOutFreq"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR2_EInvFlt"},
            {"match": "event",    "field": "Event Name",
             "value": "BUR1_EInvFlt"},
            {"match": "envbl",    "field": "EnvBl Id",
             "value": "EG_BUR3"},
            {"match": "envbl",    "field": "EnvBl Id",
             "value": "EG_BUR2"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "auxiliary converter"},
            {"match": "contains", "field": "Dist Text",
             "value": "Lifesign from B"},
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
            "BUR internal output/inverter fault — NOT a communication fault. "
            "Do not check fibre optic or Card 1302-1 first (that is for lifesign loss). "
            "BUR3:0021 No Output Frequency: BUR3 charge converter section has failed internally. "
            "BUR2:0002 Inverter fault: BUR2 drive electronics fault. "
            "Identify which BUR from the event prefix (BUR1/BUR2/BUR3). "
            "Isolate that BUR using its MCB: 127.22/1 (SB-1) for BUR1, "
            "127.22/2 (SB-2) for BUR2, 127.22/3 (SB-2) for BUR3. "
            "Inspect BUR card cage — check Card 1302-1 (control), Card 2000-140 "
            "(battery charger control), and Card 1703 (thyristor driver). "
            "BUR3 output fault specifically: check the output winding connections "
            "before replacing cards — loose connection can cause spurious no-output reading. "
            "If fault clears after MCE reset and does not recur: monitor. "
            "If it fires on every power-on cycle: card replacement needed."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN: TRAFO_OIL_BOTH
    # Standalone trigger for CCUO:0082 — both trafo oil circuits disturbed.
    # Distinct from TRAFO_OIL (one circuit, temperature trip) and
    # TRAFO_PUMP_MCB (MCB open precursor). Fires when both circuits fail
    # simultaneously without the single-circuit precursor being logged first.
    # Confirmed on IRPRP37602 (12-Apr-2026, manual register).
    # ------------------------------------------------------------------
    {
        "chain_id":    "TRAFO_OIL_BOTH",
        "name":        "Both Transformer Oil Circuits Disturbed → Emergency Shutdown",
        "subsystem":   "Transformer Cooling Circuit",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0082"},
            {"match": "contains", "field": "Dist Text",
             "value": "Dist. both trafo oil circuits"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "trafo oil"},
            {"match": "contains", "field": "Dist Text",
             "value": "temperature"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
            {"match": "contains", "field": "Dist Text",
             "value": "SPIF"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Both transformer oil circuits disturbed simultaneously — more serious than "
            "single circuit fault. Both oil pumps or both oil cooler blower circuits have "
            "failed or tripped. "
            "Step 1: Check both pump MCBs 62.1/1 (HB1) and 62.1/2 (HB2) — if both tripped, "
            "check for earth fault on oil pump busbar before resetting. "
            "Step 2: Check both oil cooler blower MCBs 59.1/1 and 59.1/2. "
            "Step 3: Check BUR output voltage — simultaneous failure of both circuits often "
            "indicates an upstream BUR output fault causing both HB-1 and HB-2 supply loss. "
            "Step 4: Check transformer oil level in expansion tanks — critically low oil level "
            "can cause thermal protection on both circuits to trigger simultaneously. "
            "Do NOT reset and run until root cause is identified — running without adequate "
            "transformer cooling risks transformer winding damage."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 25: HB-1 Busbar MCB Cluster (Multiple MCBs from same cubicle)
    # Confirmed on: 33586 (CCUO:0117-0123, 7 MCBs all from HB-1, same dates)
    # When 3+ MCBs from the same busbar (HB-1 or HB-2) trip on the same
    #   session, this is NOT 7 independent faults. It is a single upstream
    #   supply fault causing cascaded MCB trips across the busbar.
    # If reset individually the fault WILL recur. Must identify upstream cause.
    # Individual MCB codes: 0117 oil cooler blower, 0118 MR blower,
    #   0119 TM1 blower, 0120 MR scav blower, 0121 conv1 pump/fan,
    #   0122 transformer pump 1, 0123 scav oil cooler
    # ------------------------------------------------------------------
    {
        "chain_id":    "HB1_MCB_CLUSTER",
        "name":        "HB-1 Busbar MCB Cluster → Common Supply Fault",
        "subsystem":   "Auxiliary Supply (HB-1 Busbar)",
        "dcu_aware":   False,
        "max_window":  60,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0117"},
            {"match": "contains", "field": "Dist Text",
             "value": "Oil cooler blower MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0118"},
            {"match": "contains", "field": "Dist Text",
             "value": "MR blower MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0119"},
            {"match": "contains", "field": "Dist Text",
             "value": "TM 1 blower MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0120"},
            {"match": "contains", "field": "Dist Text",
             "value": "MR scav. blower MCB open"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0121"},
            {"match": "contains", "field": "Dist Text",
             "value": "Convert.1 pump or fan MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0122"},
            {"match": "contains", "field": "Dist Text",
             "value": "Transformer pump 1 MCB open"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0123"},
            {"match": "contains", "field": "Dist Text",
             "value": "Scav. oil cooler MCB open"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS06"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Multiple MCBs from HB-1 busbar tripped — this is a busbar supply fault, "
            "NOT individual blower/pump failures. "
            "DO NOT reset individual MCBs one by one — they will all trip again. "
            "Step 1: Check the main HB-1 busbar supply voltage (HBB1 cubicle, SB-1). "
            "Step 2: Check the BUR output feeding HB-1 — if BUR output is low or absent "
            "it will cause downstream MCB trips across the whole busbar. "
            "Step 3: Check for earth fault on HB-1 wiring before resetting any MCB. "
            "Measure insulation resistance on the HB-1 busbar section. "
            "Step 4: Only after confirming supply voltage is healthy and no earth fault: "
            "reset MCBs in sequence and observe which trips first — that is the actual fault. "
            "If all hold after reset: the root cause was a transient supply dip (OHE or BUR). "
            "Persistent re-trip after confirmed healthy supply: inspect motor windings "
            "on the first MCB to trip."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 26: Pantograph Mechanical Bounce
    # Confirmed on: 33586 (L:0790 x85 events), 33792 (L:0790 x64 events)
    # ECode 303E (DCU1 panto), 403E (DCU2 panto)
    # Event: D_PnBoL_CON18, D_PnBoL_CON28
    # This is a MECHANICAL fault (pantograph spring/horn wear) not a card fault.
    # High frequency bouncing causes OHE arcing, carbon contamination of VCB,
    #   and intermittent traction loss. The M2HR consistently records this as
    #   a persistent fault requiring panto maintenance.
    # ------------------------------------------------------------------
    {
        "chain_id":    "PANTO_BOUNCE",
        "name":        "Pantograph Mechanical Bouncing → OHE Contact Loss",
        "subsystem":   "Pantograph / OHE Interface (Mechanical)",
        "dcu_aware":   False,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "Pantograph bouncing"},
            {"match": "contains", "field": "Dist Text",
             "value": "L:0790"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PnBoL_CON18"},
            {"match": "event",    "field": "Event Name",
             "value": "D_PnBoL_CON28"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "303E"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "403E"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Primary voltage"},
            {"match": "contains", "field": "Dist Text",
             "value": "OHE"},
            {"match": "contains", "field": "Dist Text",
             "value": "Catenary"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "MCE off - pan was down"},
            {"match": "contains", "field": "Dist Text",
             "value": "VCB will not open"},
            {"match": "event",    "field": "Event Name",
             "value": "MPV_EPgDown10Min"},
        ],
        "severity":    "MEDIUM",
        "action":      (
            "Pantograph bouncing — mechanical fault, not an electronics fault. "
            "Inspect pantograph pan (carbon strip wear, spring tension, horn condition). "
            "Check pantograph frame for loose joints or worn pivot bearings. "
            "Check pan strip thickness — minimum 20mm, replace if less. "
            "Inspect overhead contact wire condition at the section where bouncing occurs "
            "(report to TRD if OHE stagger is excessive). "
            "High frequency bouncing (20+ events per session): pan strip replacement needed "
            "and pantograph spring tension check. "
            "Secondary risk: repeated OHE arcing contaminates VCB contacts — "
            "inspect VCB after persistent pantograph bouncing is resolved. "
            "Loco can continue in service at reduced speed if VCB is healthy, "
            "but schedule pantograph maintenance at next INSP."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 27: Line Converter HW Error / System Change Fault
    # Confirmed on: 33792 (L:0834 HW error device, L:0852 system change
    #   not successful, both on TC-2, 21-May, ECode 406D/407F)
    # EnvBl: S_CvDgEnvGrL_2 (DCU2 line converter diagnostics group)
    # L:0834 = hardware error on a device in the line converter
    # L:0852 = system change command failed (firmware/parameter mismatch,
    #   or hardware didn't respond to mode change during operation)
    # These two appearing together on TC-2 same session = single root cause.
    # ------------------------------------------------------------------
    {
        "chain_id":    "LINE_CONV_HW_FAULT",
        "name":        "Line Converter HW Error → System Change Failure",
        "subsystem":   "Line Converter (DCU2 / TC-2)",
        "dcu_aware":   True,
        "max_window":  30,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "L:0834"},
            {"match": "contains", "field": "Dist Text",
             "value": "HW error device"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "406D"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "306D"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "L:0852"},
            {"match": "contains", "field": "Dist Text",
             "value": "system change not successful"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "407F"},
            {"match": "ecode",    "field": "ECode 0",
             "value": "307F"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Iso Request CON"},
            {"match": "contains", "field": "Dist Text",
             "value": "SS01"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power Off MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Line converter hardware fault (TC-2 / DCU2). "
            "L:0834 HW error device: a hardware component in the DCU2 line converter "
            "has responded with an error — typically a board-level fault. "
            "L:0852 System change not successful: the line converter failed to complete "
            "a mode transition (startup, regenerative braking, etc.). "
            "These two appearing together = single hardware root cause. "
            "Identify DCU origin: ECode prefix 3xxx=DCU1, 4xxx=DCU2. "
            "Inspect line converter card cage on the identified DCU. "
            "Check DCU2 connector integrity — intermittent connector faults cause "
            "both HW errors and failed system changes. Reseat all connectors first. "
            "If L:0852 appears after a card replacement: firmware version mismatch — "
            "verify the replacement card matches the loco parameter spec. "
            "If no recent replacement and connectors are clean: "
            "check Line Converter Control board (CON2-A101) for damage."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN 28: VCB Will Not Close → SS01 Supply Loss
    # Complement to VCB_STUCK_ON (chain 2). That chain handles VCB stuck
    #   in the OPEN position (won't open on command). This chain handles
    #   VCB stuck in CLOSED position (won't close to restore supply).
    # Observed from M2HR fleet analysis (8 occurrences across fleet).
    # Different mechanical failure mode — different investigation procedure.
    # ------------------------------------------------------------------
    {
        "chain_id":    "VCB_NO_CLOSE",
        "name":        "VCB Will Not Close → Main Power Supply Loss",
        "subsystem":   "Main Power / VCB",
        "dcu_aware":   False,
        "max_window":  20,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "VCB will not close"},
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0115"},
            {"match": "event",    "field": "Event Name",
             "value": "MC_EMCBStkOff"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Primary voltage below minimum"},
            {"match": "contains", "field": "Dist Text",
             "value": "OHE"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01 main power off"},
            {"match": "contains", "field": "Dist Text",
             "value": "MCE off"},
        ],
        "severity":    "HIGH",
        "action":      (
            "VCB will not close — distinct from VCB stuck ON. "
            "VCB is failing to CLOSE on command, not failing to open. "
            "Check: (1) OHE voltage present at pantograph — if OHE is absent, "
            "VCB correctly refuses to close. Check OHE status first. "
            "(2) VCB pneumatic supply — closing coil requires adequate air pressure. "
            "Check MR pressure (must be > 6 kg/cm²). "
            "(3) VCB closing coil and associated relay — check SB-1 for coil fault. "
            "(4) Interlock circuit — VCB won't close if earthing switch is in. "
            "Check earthing switch (Pos. 8) is correctly withdrawn. "
            "(5) If OHE present, MR healthy, and interlocks clear: "
            "VCB closing mechanism is mechanically faulty — physical inspection required. "
            "Do not attempt more than 3 closing attempts — repeated operation "
            "of a faulty VCB risks coil burnout."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN: PRIMARY_OVERCURRENT
    # ECode: 5081 (CCUO:0130), Prio 1, 15/15 locos in fleet data
    # Mechanism: primary current exceeds maximum — OCR-78 relay trips,
    #   VCB power supply interrupted, SS01 isolation follows.
    #   Most commonly caused by OHE transient, short circuit, or
    #   line converter hardware fault. Correlation with OHE events
    #   on same section is key to distinguishing infrastructure vs hardware.
    # ------------------------------------------------------------------
    {
        "chain_id":    "PRIMARY_OVERCURRENT",
        "name":        "Primary Overcurrent → SS01 Isolation",
        "subsystem":   "Main Power (SS01)",
        "dcu_aware":   False,
        "max_window":  5,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0130"},
            {"match": "contains", "field": "Dist Text",
             "value": "Primary current above maximum"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01 main power off"},
            {"match": "contains", "field": "Dist Text",
             "value": "VCB"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "SS01 main power off"},
            {"match": "contains", "field": "Dist Text",
             "value": "Power on of MCE"},
        ],
        "severity":    "HIGH",
        "action":      (
            "Primary current exceeded maximum — OCR-78 relay has tripped. "
            "Step 1: Check OHE voltmeter — if OHE was fluctuating or absent, "
            "this is an infrastructure event; correlate with other locos on same section. "
            "Step 2: Reset VCB once (BLDJ). If it holds, resume cautiously and monitor. "
            "Step 3: If OCR trips again immediately: do NOT reset again — "
            "this is a hardware fault (line converter short circuit or IGBT failure). "
            "Check line converter cards on the affected DCU. "
            "Step 4: If both DCUs show overcurrent simultaneously: OHE fault is most likely — "
            "report to TRD and wait for OHE rectification before resuming. "
            "Do not perform more than 2 reset attempts — repeated overcurrent trips "
            "risk damage to line converter IGBTs and VCB contacts."
        ),
    },

    # ------------------------------------------------------------------
    # CHAIN: SPEEDOMETER_FAULT
    # ECode: 508F (CCUO:0144), Prio 2, confirmed on IRPRP43570 and observed
    # on additional locos in the fleet (per manual ATIL register cross-check).
    # Mechanism: speedometer signal failure — affects speed display and
    # speed-dependent protection functions (overspeed, vigilance).
    # ------------------------------------------------------------------
    {
        "chain_id":    "SPEEDOMETER_FAULT",
        "name":        "Speedometer Failed → Speed Display/Protection Affected",
        "subsystem":   "VCI / Speed Sensing",
        "dcu_aware":   False,
        "max_window":  10,
        "trigger": [
            {"match": "contains", "field": "Dist Text",
             "value": "CCUO:0144"},
            {"match": "contains", "field": "Dist Text",
             "value": "Speedometer failed"},
        ],
        "propagation": [
            {"match": "contains", "field": "Dist Text",
             "value": "Speed sensor"},
            {"match": "contains", "field": "Dist Text",
             "value": "TM1-Bogie"},
            {"match": "contains", "field": "Dist Text",
             "value": "TM2-Bogie"},
        ],
        "terminal": [
            {"match": "contains", "field": "Dist Text",
             "value": "Speedometer"},
        ],
        "severity":    "MEDIUM",
        "action":      (
            "Speedometer signal failure — affects cab speed display and "
            "speed-dependent protection functions (overspeed protection, vigilance control, "
            "regenerative brake speed threshold). "
            "Step 1: Check which axle's speed sensor (SS01/SS02) is implicated — "
            "cross-reference with any TM isolation events occurring shortly after. "
            "Step 2: Inspect speed sensor wiring and connector at the affected axle box. "
            "Step 3: Check sensor air gap if accessible — speed sensors are sensitive to "
            "mounting clearance drift. "
            "Step 4: If fault clears on next power-on and does not recur: likely a transient "
            "signal dropout, monitor only. "
            "Step 5: If fault is persistent: loco must not be used for high-speed running until "
            "resolved — speed-dependent protections may not function correctly. "
            "Note: distinct from individual traction motor speed sensor faults reported via "
            "VCI:0023 (TM_MOTOR_ISOLATED chain) — CCUO:0144 is the consolidated cab "
            "speedometer display fault."
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