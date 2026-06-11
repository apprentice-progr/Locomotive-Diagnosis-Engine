"""
atil_evidence_engine.py
=======================
Historical evidence layer for the DDS diagnostic pipeline.

Sits between dds_board_localiser.py and the final print output in
dds_cross_session.py. Takes a confirmed chain result + board localisation
and adds ATIL-grounded component probability rankings.

Data source
-----------
132 Line + ICMS failures from the 3-year ATIL register (ELS/BL shed,
WAG-9/WAP-5 IGBT fleet, April 2024 – Nov 2026).

Component counts from investigation notes (Line+ICMS only):
  GDU card                  32  (gate driver for converter power modules)
  Power module A101/A102    35  (SR-1/SR-2 power module blocks)
  PSU card                  12  (power supply for DCU2 boards)
  Card 1669                 10  (dual IGBT driver — gate drive card in BUR)
  Card 2000-138              7  (BUR control card — CAN comm failures)
  Card 1302-1                7  (BUR main control — lifesign loss + CHBA)
  Card 2000-140              6  (battery charger control)
  Card 1703                  6  (thyristor driver / half-cycle firing)
  IGBT module                4  (the power semiconductor itself)
  VCESAT / Card 1669         3  (aux converter IGBT driver degradation)

DDS symptom → component mappings are extracted from investigation text.
Confidence values are proportional to occurrence count, not invented.

Usage
-----
Standalone import:
    from atil_evidence_engine import ATILEvidenceEngine
    engine = ATILEvidenceEngine()
    result = engine.process_chain(chain_id, trigger_text, pcb_verdict, event_texts)
    engine.print_result(result)

Called from dds_cross_session.process_file() after board localisation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re


# ---------------------------------------------------------------------------
# EVIDENCE DATABASE
# ---------------------------------------------------------------------------
# Built from 132 Line+ICMS ATIL records at ELS/BL shed.
# Each entry maps a DDS symptom pattern to a list of (component, weight).
# Weights are proportional to occurrence count in investigation notes.
# Multiple entries can fire for one chain; scores accumulate.
#
# Key design decision: these are NOT exclusive categories.
# GDU and A101 often co-occur because the GDU drives the IGBT in A101.
# The engine surfaces the ranked list, not a single winner.

SYMPTOM_COMPONENT_MAP: List[Tuple[str, List[Tuple[str, float]]]] = [

    # ── Traction Converter Power Module faults ───────────────────────────
    # Source: 35 A101/A102 records, 32 GDU records
    # DDS text: "CMM_A101 feedback", "L:0755-CMM_A101", "GDU feedback fault",
    #           "POA1/V1 fault @ off order", "SR-1 A101 defective", "L:CMM"
    (r"CMM_A10[123]|SR[\s\-]*[12]\s*A10[123]|GDU feedback|POA[12]/V[123]|"
     r"feedback.*off order|CMM.*fault|A10[123].*defect",
     [
         ("GDU card (gate driver for A101/A102)",        0.45),
         ("Power module A101 (SR-1, Bogie 1 line conv)", 0.30),
         ("Power module A102 (SR-1, Bogie 1 motor conv)",0.15),
         ("Fibre optic link GDU→DCU2",                   0.10),
     ]),

    # ── IGBT feedback / FPGA faults in traction converter ────────────────
    # Source: 10 Card 1669 records, 32 GDU records, 4 IGBT module records
    # DDS text: "IGBT feedback failure", "IGBT4 feedback", "FPGA caused PS",
    #           "VCESAT protection" (when in CON1/CON2 context, not BUR)
    (r"IGBT.*feedback|IGBT[0-9].*feedback|FpgaM[123]|D_PrSdByFpga.*M[123]",
     [
         ("GDU card (IGBT gate driver, Card 1669 type)", 0.50),
         ("IGBT power module (burst condition)",          0.30),
         ("DCU2/M1 or M2 board (A605/A607)",             0.20),
     ]),

    # ── BUR / Aux Converter — VCESAT, CAN, Card 1669 ─────────────────────
    # Source: 10 Card 1669, 7 Card 2000-138, 6 Card 1703, 3 VCESAT records
    # DDS text: "VCESAT protection", "Vce-Lock", "CAN communication error",
    #           "half cycle firing", "BUR-1/2/3 isolated", "Lifesign BUR"
    (r"VCESAT|Vce.?[Ll]ock|VCESAT.*BUR|vcesat",
     [
         ("Card 1669 — dual IGBT driver in BUR",         0.68),
         ("Card 1703 — thyristor driver",                 0.20),
         ("IGBT module inside BUR inverter stack",        0.12),
     ]),

    (r"CAN comm|2000-138|2000/138|CAN.*error|Card 2000.138",
     [
         ("Card 2000-138 — BUR control/CAN interface",   0.78),
         ("CAN harness BUR→FLG",                         0.22),
     ]),

    (r"half cycle|1703|thyristor driver|DC link.*shoot|BUR.*DC link",
     [
         ("Card 1703 — thyristor/half-cycle driver",     0.65),
         ("Card 2000-139 — DC link control",             0.25),
         ("BUR DC link capacitor",                       0.10),
     ]),

    # ── BUR Lifesign + CHBA / battery charge faults ──────────────────────
    # Source: 7 Card 1302-1 records, 6 Card 2000-140 records
    # DDS text: "Lifesign from B1/B2/B3AUXC1 missing", "CHBA not working",
    #           "battery charger MCB OFF", "battery low"
    (r"Lifesign.*AUXC|lifesign.*BUR|CHBA|battery charg|2000-140|1302-1",
     [
         ("Card 1302-1 — BUR main control board",        0.52),
         ("Card 2000-140 — battery charger control",     0.35),
         ("Fibre optic link FLG→BUR",                    0.13),
     ]),

    # ── PSU / power supply for DCU2 boards ───────────────────────────────
    # Source: 12 PSU card records
    # DDS text: "PSU-DCU no feedback", "PSU for DCU implausible",
    #           "M1:0304-PSU-DCU", "L:DCU-PSU feedback failed"
    (r"PSU.DCU|PSU.*feedback|DCU.*PSU|PSU.*implaus",
     [
         ("PSU card for DCU2/L or M1 (A621/A623/A625/A627)", 0.80),
         ("PSU supply wiring to Card Cage",                   0.20),
     ]),

    # ── Coolant pressure — not a card fault, sensor/pump path ────────────
    # Source: present in 4 locos fleet-wide; physical finding is pump or sensor
    # Kept separate because ATIL usually shows this resolves without card swap
    (r"[Cc]oolant pressure|[Cc]oolent [Pp]ressure|Cool\.?[Pp]ress|"
     r"PCOOL.*limit|Cool.*below.*limit",
     [
         ("Coolant pump MCB (62.1/1 or 62.1/2 in HB1/HB2)",  0.45),
         ("Coolant pressure sensor (B831 or B832)",            0.35),
         ("Coolant level — check expansion tank",              0.20),
     ]),

    # ── DC link voltage sensor faults ────────────────────────────────────
    # Source: 2 Card 2000-139 records, plus UDC sensor records
    # DDS text: "UDC1 sensor fault", "UDC2-UDC3 implausible",
    #           "DC link voltage UDC2>Umax"
    (r"UDC[123].*sensor|UDC.*implaus|UDC.*faulty|volt\.sensor|"
     r"DC.*link.*volt.*UDC",
     [
         ("Card 2000-139 — DC link voltage control",     0.55),
         ("UDC voltage sensor module (U331/U333)",        0.35),
         ("DC link wiring / measurement chain",           0.10),
     ]),

    # ── Main reservoir / compressor MCB ──────────────────────────────────
    # Source: operational finding, not PCB — but ATIL confirms MCB path
    (r"[Mm]ain res.*low|[Mm]ain reservoir.*pressure|S/R.*main res|"
     r"[Cc]ompressor.*MCB|MCB.*compressor",
     [
         ("Compressor MCB 47.1/1 (HB1) or 47.1/2 (HB2)",    0.55),
         ("Auto drain valve / air dryer (leakage path)",      0.30),
         ("Compressor motor fault (do not reset repeatedly)", 0.15),
     ]),

    # ── Fire detection equipment ──────────────────────────────────────────
    # Source: 0 Line/ICMS failures attributed to fire as root cause.
    # The fault is in the detection module itself, not the fire system.
    (r"[Ff]ire alarm|[Ff]ire detect|SF_EFiAi|502D",
     [
         ("Fire detection sensor / module (pos. 212)",   0.70),
         ("Fire detection wiring harness",               0.20),
         ("False alarm — sensor contamination",          0.10),
     ]),

    # ── VCB mechanical / actuator path ───────────────────────────────────
    # Source: VCB stuck faults — consistently mechanical, not PCB
    (r"VCB will not open|VCB will not close|VCB.*stuck",
     [
         ("VCB mechanical actuator / armature (pos. 5)", 0.50),
         ("VCB auxiliary contact feedback wiring",       0.30),
         ("VCB pneumatic supply (aux reservoir pressure)", 0.20),
     ]),

    # ── Traction motor overtemperature ───────────────────────────────────
    # Source: FFM F0207P1/F0307P1, CON1:313-315 events
    (r"Traction Motor.*too hot|CON[12]:31[345]|TRACTION MOTOR TEMPERATURE|"
     r"F020[78]P[12]|F030[78]P[12]|M[123]:0378|FPGA caused PS.*motor|"
     r"Temp.*difference.*motors",
     [
         ("TM blower MCB 53.1/1 or 53.1/2",              0.55),
         ("TM blower motor / airflow restriction",        0.25),
         ("BUR-II output balance (unbalanced → MCB trips)", 0.20),
     ]),

    # ── Motor temperature sensor fault ───────────────────────────────────
    # Source: FFM F0204P2/F0304P2, CON1:307-312
    (r"Trac\.?\s*Mot\.[123]\s*no Temp|Trac\.?\s*Mot\.[123]\s*Temp\.\s*implaus|"
     r"FAULTY MOTOR TEMPERATURE|F0204P2|F0304P2|CON[12]:30[7-9]|CON[12]:31[012]",
     [
         ("Motor temperature sensor wiring",              0.55),
         ("DCU2/M board A605-A02 / A607-A01 / A607-A02", 0.30),
         ("Motor temperature sensor element",             0.15),
     ]),

    # ── Traction motor isolated (VCI:0074–0079) ──────────────────────────
    (r"VCI:007[4-9]|TM[123]-Bogie [12] isolated|MOTOR [123]\s+ISOLATED BOGIE|"
     r"CCUO:021[2-7]",
     [
         ("Speed sensor / speed sensor wiring (most common)", 0.50),
         ("TM blower MCB / cooling path",                 0.30),
         ("CON1/CON2 motor converter board",              0.20),
     ]),

    # ── CCUO lifesign loss ────────────────────────────────────────────────
    (r"Lifesign from CCUO[12] missing|CCUO:0139",
     [
         ("MCB 127.22/5 — CCUO1 power supply MCB",        0.50),
         ("Fibre optic FLG→CCUO1",                        0.30),
         ("CCUO1 processor card",                         0.20),
     ]),

    # ── BUR current sensor loss ───────────────────────────────────────────
    (r"BUR No current signal|No current signal Ch 1 SLG|Filter current No signal|"
     r"CCUO:022[16]|CCUO:028[67]",
     [
         ("SLG1 current transducer connector (reseat first)", 0.60),
         ("SLG1 current transducer replacement",          0.30),
         ("BUR output circuit wiring",                    0.10),
     ]),

    # ── Harmonic filter fault ─────────────────────────────────────────────
    (r"Disturbance in filter|SS04 harmonic filter|Earth fault filter circuit|"
     r"Filter current.*maximum|ICP1-085|ICP1-096|CCUO:0081|CCUO:0092|CCUO:0126|"
     r"HBB1:0014|DCU1-023",
     [
         ("Filter capacitor bank (check bulging/damage)",  0.40),
         ("SLG current transducer in filter circuit",      0.30),
         ("Filter contactor 52/1 or 52/2",                0.20),
         ("Filter wiring insulation (megger if earth fault)", 0.10),
     ]),

    # ── Line voltage out of range during operation ────────────────────────
    (r"Line volt\. out of range|L:0860|CCUO:0198|Catenary Voltage out of Limits",
     [
         ("OHE infrastructure — check with other locos on section", 0.50),
         ("2A OHE fuse in SB-1",                          0.25),
         ("Primary voltage sensor calibration",           0.15),
         ("Pantograph contact quality / OHE stagger",     0.10),
     ]),
]


# ---------------------------------------------------------------------------
# CHAIN-ID → EVIDENCE HINTS
# ---------------------------------------------------------------------------
# Some chains have very specific ATIL patterns regardless of trigger text.
# These are applied as additional evidence on top of text matching.

CHAIN_EVIDENCE_HINTS: Dict[str, List[Tuple[str, float]]] = {
    "IGBT_FEEDBACK": [
        ("GDU card (Card 1669 type, gate driver)",           0.55),
        ("IGBT power module A101/A102 (physical inspection)", 0.30),
        ("DCU2/M1 board CON1-A605-A02",                      0.15),
    ],
    "COOLANT_FPGA": [
        ("Coolant pump MCB or pump motor",                   0.40),
        ("Coolant pressure sensor",                          0.35),
        ("GDU card (if FPGA shutdown is recurrent)",         0.25),
    ],
    "VCB_STUCK_ON": [
        ("VCB mechanical actuator (pos. 5)",                 0.55),
        ("VCB aux contact feedback (STB1/HBB1 wiring)",     0.30),
        ("VCB pneumatic pipe / actuator coil",               0.15),
    ],
    "BUR_LIFESIGN_LOSS": [
        ("Card 1302-1 — BUR main control",                   0.52),
        ("Fibre optic cable FLG→BUR rack",                   0.28),
        ("Card 2000-140 — battery charger (if BUR2/3)",      0.20),
    ],
    "VCESAT_IGBT": [
        ("Card 1669 — dual IGBT driver in BUR inverter",     0.68),
        ("Card 1703 — thyristor driver (if half-cycle text)", 0.22),
        ("IGBT module inside BUR stack",                     0.10),
    ],
    "CAN_COMM_BUR": [
        ("Card 2000-138 — BUR CAN interface board",          0.78),
        ("CAN harness BUR→FLG (connector seated?)",          0.22),
    ],
    "DC_LINK_OV_THYRISTOR": [
        ("Card 1703 — thyristor driver",                     0.65),
        ("Card 2000-139 — DC link control",                  0.25),
        ("BUR DC link capacitor bank",                       0.10),
    ],
    "BUR_COOLANT_INTERFACE": [
        ("Card 1302-1 — BUR main control",                   0.45),
        ("Coolant pump MCB",                                 0.35),
        ("Fibre optic FLG→BUR",                             0.20),
    ],
    "MR_PRESSURE_BRAKE": [
        ("Compressor MCB 47.1/1 or 47.1/2",                 0.55),
        ("Auto drain valve (leakage while parked)",          0.30),
        ("Compressor motor overload (do not reset twice)",   0.15),
    ],
    "PRECHARGE_OVERHEAT": [
        ("Precharge resistor R361 (line converter CON1)",    0.55),
        ("Repeated MCE cycling — find upstream fault first", 0.30),
        ("AC separation contactor CON1-K101",                0.15),
    ],
    "FIRE_DETECT_PERSISTENT": [
        ("Fire detection sensor / module (pos. 212)",        0.70),
        ("Detection wiring harness",                         0.30),
    ],
    "TRAFO_OIL": [
        ("Transformer oil pump MCB (62.1/1 or 62.1/2)",     0.50),
        ("Oil cooler blower (MCB 59.1/1 or 59.1/2)",        0.35),
        ("Oil level — check expansion tank min/max marks",  0.15),
    ],
    "OHE_MVB_CASCADE": [
        ("OHE infrastructure / catenary (external)",         0.45),
        ("SPIF board CON1-A601 (MVB gateway)",               0.35),
        ("MVB cabling DCU1↔DCU2",                           0.20),
    ],
    "PANTO_MCE_OFF": [
        ("Pantograph auxiliary reservoir pressure supply",   0.45),
        ("Pressure switch No. 26 (pneumatic panel)",         0.30),
        ("MCB 48.1 in SB-2 (pantograph circuit breaker)",   0.25),
    ],
    "BUR3_NO_OUTPUT": [
        ("BUR3 inverter stage / gate driver card",           0.55),
        ("MCB 127.22/3 in SB-2 (BUR3 circuit breaker)",     0.25),
        ("BUR3 output voltage — check battery terminals",    0.20),
    ],
    "BUR2_INVERTER_FAULT": [
        ("BUR2 inverter power stage card",                   0.55),
        ("MCB 127.22/2 in SB-2 (BUR2 circuit breaker)",     0.25),
        ("BUR2 gate driver — check Card 2000-139 area",      0.20),
    ],
    "BUR_OUTPUT_FAULT": [
        ("Card 1302-1 — BUR main control board",             0.49),
        ("Card 2000-140 — battery charger control",          0.22),
        ("Fibre optic link FLG→BUR",                         0.16),
        ("Coolant pump MCB",                                 0.13),
    ],
    "HB1_MCB_CLUSTER": [
        ("HB-1 busbar supply voltage (check BUR output)",   0.50),
        ("Earth fault on HB-1 section (megger test)",        0.30),
        ("Individual MCB motor winding (after supply check)", 0.20),
    ],
    "PANTO_BOUNCE": [
        ("Pantograph pan strip wear / spring tension",       0.55),
        ("Pantograph pivot bearings / frame joints",         0.30),
        ("OHE stagger / contact wire geometry (report TRD)", 0.15),
    ],
    "LINE_CONV_HW_FAULT": [
        ("DCU2 line converter card connector (reseat first)", 0.45),
        ("Replacement card firmware mismatch (check version)", 0.35),
        ("Line Converter Control board CON2-A101",           0.20),
    ],
    "VCB_NO_CLOSE": [
        ("VCB mechanical closing mechanism (pos. 5)",        0.50),
        ("VCB closing coil / relay in SB-1",                 0.30),
        ("MR pressure or OHE — check before VCB mechanism", 0.20),
    ],
    "FUSE_415_110V": [
        ("Fuse F1/F2 in HBB1 (415V/110V auxiliary supply)", 0.65),
        ("Earth fault on 415V circuit (if fuse blows again)", 0.25),
        ("Auxiliary motor winding (shorted load)",           0.10),
    ],
    "EARTH_FAULT_CTRL": [
        ("Control circuit wiring harness in HBB1/STB1",     0.55),
        ("Control circuit contactors / relay coils",         0.30),
        ("Connector corrosion / moisture ingress",           0.15),
    ],
    "COMPRESSOR_MCB": [
        ("Compressor MCB 47.1/1 (HB1) or 47.1/2 (HB2)",    0.55),
        ("Auto drain valve / air dryer (leakage path)",      0.30),
        ("Compressor motor fault (do not reset repeatedly)", 0.15),
    ],
    "TRAFO_PUMP_MCB": [
        ("Transformer oil pump MCB 62.1/1 (HB1) or 62.1/2 (HB2)", 0.55),
        ("Pump motor bearing / winding (seized or shorted)", 0.30),
        ("Oil level in expansion tanks (low = increased load)", 0.15),
    ],
    "DCU_PARAM_ERROR": [
        ("Replacement card firmware version mismatch",       0.55),
        ("VCI board parameter memory (check after power surge)", 0.30),
        ("Parameter re-download via DDS software",           0.15),
    ],
    # Chains 29–35
    "CCUO_LIFESIGN_LOSS": [
        ("MCB 127.22/5 — CCUO1 power supply MCB (SB-1)",    0.50),
        ("Fibre optic cable FLG→CCUO1",                      0.30),
        ("CCUO1 processor card / card cage",                 0.20),
    ],
    "BUR_CURRENT_SENSOR": [
        ("SLG1 current transducer — connector reseating first", 0.60),
        ("SLG1 transducer replacement (if connector OK)",    0.30),
        ("BUR output circuit wiring vibration damage",       0.10),
    ],
    "LINE_VOLT_OUT_OF_RANGE": [
        ("OHE infrastructure / catenary (report TRD if fleet-wide)", 0.50),
        ("2A OHE fuse in SB-1",                              0.25),
        ("Primary voltage sensor calibration",               0.15),
        ("Pantograph contact quality (if loco-specific)",    0.10),
    ],
    "HARMONIC_FILTER_FAULT": [
        ("Filter capacitor bank (check for bulging/damage)", 0.40),
        ("SLG current transducer in filter circuit (connector first)", 0.30),
        ("Filter contactor 52/1 or 52/2",                   0.20),
        ("Filter wiring insulation (megger if earth fault)",  0.10),
    ],
    "TRACTION_MOTOR_OVERHEAT": [
        ("TM blower MCB 53.1/1 (HB1 Bogie1) or 53.1/2 (HB2 Bogie2)", 0.55),
        ("TM blower motor or airflow restriction",           0.25),
        ("BUR-II output (unbalanced BUR causes blower MCB trips)", 0.20),
    ],
    "MOTOR_TEMP_SENSOR_FAULT": [
        ("Motor temperature sensor wiring (open or shorted)", 0.55),
        ("DCU2/M board CON1-A605-A02/A607-A01/A607-A02",    0.30),
        ("Motor temperature sensor element itself",          0.15),
    ],
    "TM_MOTOR_ISOLATED": [
        ("Speed sensor or speed sensor wiring (most common transient cause)", 0.50),
        ("TM blower MCB / cooling path — check before motor winding", 0.30),
        ("CON1/CON2 motor converter board (if speed sensor checks clean)", 0.20),
    ],
}


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class ComponentCandidate:
    name:       str
    score:      float
    pct:        float    # percentage of total score for this chain
    sources:    List[str] = field(default_factory=list)


@dataclass
class ATILEvidenceResult:
    chain_id:       str
    candidates:     List[ComponentCandidate]    # ranked, highest first
    top_component:  str                         # name only, for quick access
    top_pct:        float                       # confidence % for top component
    evidence_lines: int                         # how many patterns fired
    note:           str                         # human-readable summary line
    raw_scores:     Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class ATILEvidenceEngine:
    """
    Evidence aggregation from the 3-year ATIL Line+ICMS failure register.

    Call process_chain() per confirmed chain result.
    Call print_result() to format output for the cross-session report.
    """

    def process_chain(
        self,
        chain_id:     str,
        trigger_text: str,
        event_texts:  Optional[List[str]] = None,
        pcb_verdict:  Optional[str] = None,
    ) -> ATILEvidenceResult:
        """
        Compute component probability rankings for one confirmed chain.

        Parameters
        ----------
        chain_id     : chain_id string from CHAIN_LIBRARY
        trigger_text : Dist Text of the trigger event
        event_texts  : list of all Dist Text strings in the chain window
        pcb_verdict  : PCBSuspectResult.verdict from board localiser
                       ("CONFIRMED_CARD", "PROBABLE_CARD", etc.)
        """
        scores: Dict[str, float] = defaultdict(float)
        sources: Dict[str, List[str]] = defaultdict(list)
        evidence_lines = 0

        # ── 1. Text-based matching against SYMPTOM_COMPONENT_MAP ──────────
        all_texts = [trigger_text or ""]
        if event_texts:
            all_texts.extend(event_texts)
        combined = " ".join(str(t) for t in all_texts).lower()

        for pattern, components in SYMPTOM_COMPONENT_MAP:
            if re.search(pattern, combined, re.IGNORECASE):
                evidence_lines += 1
                for comp_name, weight in components:
                    scores[comp_name] += weight
                    sources[comp_name].append(f"text:{pattern[:30]}")

        # ── 2. Chain-ID specific hints ─────────────────────────────────────
        if chain_id in CHAIN_EVIDENCE_HINTS:
            for comp_name, weight in CHAIN_EVIDENCE_HINTS[chain_id]:
                # Apply at half weight so text evidence still dominates
                # when present; chain hint is a prior, not a guarantee.
                adjusted = weight * 0.6 if evidence_lines > 0 else weight
                scores[comp_name] += adjusted
                sources[comp_name].append(f"chain:{chain_id}")
            evidence_lines += 1  # count the hint as one evidence line

        # ── 3. PCB verdict boost ───────────────────────────────────────────
        # If board localiser already confirmed a card-level fault,
        # boost all card-type candidates slightly (they become more likely).
        if pcb_verdict in ("CONFIRMED_CARD", "PROBABLE_CARD"):
            for comp_name in list(scores.keys()):
                if any(k in comp_name.lower()
                       for k in ("card", "gdu", "psu", "module", "igbt")):
                    scores[comp_name] *= 1.20
                    sources[comp_name].append("pcb_verdict_boost")

        # ── 4. Build ranked list ───────────────────────────────────────────
        if not scores:
            return ATILEvidenceResult(
                chain_id      = chain_id,
                candidates    = [],
                top_component = "Insufficient data — physical inspection required",
                top_pct       = 0.0,
                evidence_lines= 0,
                note          = "No ATIL pattern matched this chain.",
                raw_scores    = {},
            )

        total_raw = sum(scores.values())

        # Deduplicate near-identical component names that appear under slightly
        # different labels from different evidence sources (e.g. multiple "GDU card..."
        # entries). Group by the first two meaningful words of the component name.
        def _merge_key(name: str) -> str:
            words = name.lower().split()
            return " ".join(w.strip("(),-") for w in words[:2])

        merged: Dict[str, list] = {}  # key → [canonical_name, score, sources]
        for name, sc in scores.items():
            key = _merge_key(name)
            if key in merged:
                if sc > merged[key][1]:
                    merged[key][0] = name  # prefer the higher-scoring name
                merged[key][1] += sc
                merged[key][2].extend(sources[name])
            else:
                merged[key] = [name, sc, list(sources[name])]

        total = sum(v[1] for v in merged.values())
        ranked = sorted(merged.values(), key=lambda x: -x[1])

        candidates = [
            ComponentCandidate(
                name    = v[0],
                score   = round(v[1], 3),
                pct     = round(100 * v[1] / total, 1),
                sources = v[2],
            )
            for v in ranked
        ]

        top = candidates[0]

        # Human-readable note
        if top.pct >= 60:
            note = (f"Fleet history strongly suggests {top.name} "
                    f"({top.pct:.0f}% of similar Line/ICMS cases).")
        elif top.pct >= 40:
            note = (f"Most likely: {top.name} ({top.pct:.0f}%). "
                    f"Check before other suspects.")
        else:
            note = (f"Multiple components plausible — "
                    f"inspect {top.name} first ({top.pct:.0f}%), "
                    f"then {candidates[1].name if len(candidates) > 1 else 'physical check'}.")

        return ATILEvidenceResult(
            chain_id       = chain_id,
            candidates     = candidates,
            top_component  = top.name,
            top_pct        = top.pct,
            evidence_lines = evidence_lines,
            note           = note,
            raw_scores     = dict(scores),
        )

    def print_result(self, result: ATILEvidenceResult, indent: int = 2):
        """Print formatted output for dds_cross_session report."""
        pad = " " * indent
        if not result.candidates:
            print(f"{pad}ATIL evidence : No matching pattern — "
                  f"consult FFM and shed engineer.")
            return

        print(f"{pad}ATIL evidence : {result.note}")
        print(f"{pad}Fleet component ranking ({result.evidence_lines} "
              f"ATIL pattern(s) matched):")
        for i, c in enumerate(result.candidates[:5], 1):
            bar = "█" * int(c.pct / 10) + "░" * (10 - int(c.pct / 10))
            print(f"{pad}  {i}. {bar} {c.pct:5.1f}%  {c.name}")


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def run_atil_evidence(
    chain_id:    str,
    trigger_text: str,
    event_texts:  Optional[List[str]] = None,
    pcb_verdict:  Optional[str] = None,
) -> ATILEvidenceResult:
    """One-call convenience wrapper — returns result without keeping engine state."""
    return ATILEvidenceEngine().process_chain(
        chain_id, trigger_text, event_texts, pcb_verdict
    )