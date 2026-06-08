"""
DDS Board Localiser
====================
Determines the probable physical PCB card / board responsible for a fault
using three columns that are already in every DDS export:

    ECode 0   — fault register value in hex; prefix encodes the originating
                 processor, suffix encodes the specific fault bit
    Event Name — internal software event identifier; suffix often carries
                 board-level tag (_CON15 = converter1 / _CON16 = converter2,
                 M1/M2/M3 = motor converter 1/2/3, L = line converter,
                 SPIF = Standard Propulsion Interface board)
    EnvBl Id  — which processor's environmental block was captured;
                 disagreement between EnvBl and ECode prefix signals a
                 cross-DCU cascade (MVB pullwire scenario)

Card reference source: Bombardier WAG9 FFM (3EH-214057-0001), Section 6.3.3.9
(CON1/CON2 processor disturbance forms — each lists the replacement card
for each fault type).

How to use
----------
Standalone:
    python dds_board_localiser.py <dds_excel_file>

As a module (called from dds_cross_session.py):
    from dds_board_localiser import localise_board, score_pcb_suspect, annotate_chains

The main entry point for integration is annotate_chains(), which takes the
list of LinkedChainResult objects from the cross-session tracker and adds
board localisation to each confirmed chain in-place.
"""

import sys
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# PHYSICAL BOARD MAP
# ---------------------------------------------------------------------------
# Keyed by ECode 0 hex prefix (first 2-4 chars).
# Each entry carries:
#   board_id    : internal reference matching Bombardier FFM notation
#   description : human-readable card description
#   part_ref    : FFM part reference (CON1-Axxx notation)
#   replaces    : what to physically swap if this board is confirmed faulty
#   confidence  : base confidence that this prefix → this board (0.0–1.0)
#                 Some prefixes are shared across multiple faults; lower
#                 confidence means the Event Name or EnvBl must confirm.

ECODE_BOARD_MAP: Dict[str, dict] = {
    # ── SPIF / CCUO (Central Control Unit Output) ──────────────────────────
    # 50xx–5Fxx = SPIF board on one of the converters (A601)
    # The SPIF is the MVB gateway between the vehicle bus and the traction
    # converter internal bus. Most CCUO:xxxx events originate here.
    "50": {"board_id": "SPIF",   "description": "Standard Propulsion Interface (SPIF/VCU-C)",
           "part_ref": "CON-A601", "replaces": "CON1-A601 or CON2-A601",
           "confidence": 0.7},
    "51": {"board_id": "SPIF",   "description": "Standard Propulsion Interface (SPIF/VCU-C)",
           "part_ref": "CON-A601", "replaces": "CON1-A601 or CON2-A601",
           "confidence": 0.7},
    "52": {"board_id": "SPIF",   "description": "Standard Propulsion Interface (SPIF/VCU-C)",
           "part_ref": "CON-A601", "replaces": "CON1-A601 or CON2-A601",
           "confidence": 0.65},
    "53": {"board_id": "SPIF",   "description": "Standard Propulsion Interface (SPIF/VCU-C)",
           "part_ref": "CON-A601", "replaces": "CON1-A601 or CON2-A601",
           "confidence": 0.65},
    "5071": {"board_id": "SPIF_VCB", "description": "VCB holding circuit / SPIF output",
             "part_ref": "CON-A601", "replaces": "Check VCB pos.5 before replacing A601",
             "confidence": 0.8},
    "508B": {"board_id": "SPIF_PAN", "description": "Pantograph pressure input / SPIF",
             "part_ref": "CON-A601", "replaces": "Check pneumatic circuit before A601",
             "confidence": 0.75},
    "5029": {"board_id": "SPIF_MR",  "description": "Main reservoir pressure input / SPIF",
             "part_ref": "CON-A601", "replaces": "Check compressor MCBs before A601",
             "confidence": 0.75},
    "502D": {"board_id": "SPIF_FIRE","description": "Fire detection input / SPIF",
             "part_ref": "CON-A601", "replaces": "Check fire detection unit pos.212",
             "confidence": 0.8},

    # ── DCUL — Line Converter Control board ────────────────────────────────
    # 30xx = DCU1 line side (CON1-A605-A01)
    # 40xx = DCU2 line side (CON2-A605-A01)
    "30": {"board_id": "DCUL_1",  "description": "Line Converter Control (DCU2/L) — Bogie 1",
           "part_ref": "CON1-A605-A01", "replaces": "CON1-A605-A01 (DCU2/L)",
           "confidence": 0.85},
    "40": {"board_id": "DCUL_2",  "description": "Line Converter Control (DCU2/L) — Bogie 2",
           "part_ref": "CON2-A605-A01", "replaces": "CON2-A605-A01 (DCU2/L)",
           "confidence": 0.85},

    # ── DCUM1 — Motor Converter 1 board ────────────────────────────────────
    # 31xx = CON1 motor 1 (CON1-A605-A02)
    # 41xx = CON2 motor 1 (CON2-A605-A02)
    "31": {"board_id": "DCUM1_1", "description": "Motor Converter 1 Control (DCU2/M1) — Bogie 1",
           "part_ref": "CON1-A605-A02", "replaces": "CON1-A605-A02 (DCU2/M1)",
           "confidence": 0.9},
    "41": {"board_id": "DCUM1_2", "description": "Motor Converter 1 Control (DCU2/M1) — Bogie 2",
           "part_ref": "CON2-A605-A02", "replaces": "CON2-A605-A02 (DCU2/M1)",
           "confidence": 0.9},

    # ── DCUM2/M3 — Motor Converter 2 & 3 boards ────────────────────────────
    # 32xx–33xx = CON1 motor 2 (CON1-A607-A01)
    # 36xx–37xx = CON1 motor 3 (CON1-A607-A02)
    "32": {"board_id": "DCUM2_1", "description": "Motor Converter 2 Control (DCU2/M2) — Bogie 1",
           "part_ref": "CON1-A607-A01", "replaces": "CON1-A607-A01 (DCU2/M2)",
           "confidence": 0.9},
    "42": {"board_id": "DCUM2_2", "description": "Motor Converter 2 Control (DCU2/M2) — Bogie 2",
           "part_ref": "CON2-A607-A01", "replaces": "CON2-A607-A01 (DCU2/M2)",
           "confidence": 0.9},
    "36": {"board_id": "DCUM3_1", "description": "Motor Converter 3 Control (DCU2/M3) — Bogie 1",
           "part_ref": "CON1-A607-A02", "replaces": "CON1-A607-A02 (DCU2/M3)",
           "confidence": 0.9},
    "46": {"board_id": "DCUM3_2", "description": "Motor Converter 3 Control (DCU2/M3) — Bogie 2",
           "part_ref": "CON2-A607-A02", "replaces": "CON2-A607-A02 (DCU2/M3)",
           "confidence": 0.9},

    # ── PSU boards ─────────────────────────────────────────────────────────
    # Power Supply Units — one per DCU2 board
    "34": {"board_id": "PSU_L1",  "description": "PSU for DCUL — Bogie 1",
           "part_ref": "CON1-A621", "replaces": "CON1-A621 (PSU/L)",
           "confidence": 0.75},
    "35": {"board_id": "PSU_M1_1","description": "PSU for DCUM1 — Bogie 1",
           "part_ref": "CON1-A623", "replaces": "CON1-A623 (PSU/M1)",
           "confidence": 0.75},
    "38": {"board_id": "PSU_M2_1","description": "PSU for DCUM2 — Bogie 1",
           "part_ref": "CON1-A625", "replaces": "CON1-A625 (PSU/M2)",
           "confidence": 0.75},
    "39": {"board_id": "PSU_M3_1","description": "PSU for DCUM3 — Bogie 1",
           "part_ref": "CON1-A627", "replaces": "CON1-A627 (PSU/M3)",
           "confidence": 0.75},

    # ── Exact 4-char entries derived from actual DDS data ─────────────────────
    # BUR lifesign — ECode 5009/500A/500B, EnvBl EG_FLG → CCUO sees BUR MVB dropout
    "5009": {"board_id": "BUR1_COMM",  "description": "BUR1 MVB communication / CHBA Interface",
             "part_ref": "BUR1-rack",  "replaces": "Card 2000-139 or fibre optic to BUR1 rack",
             "confidence": 0.92},
    "500A": {"board_id": "BUR2_COMM",  "description": "BUR2 MVB communication / CHBA Interface",
             "part_ref": "BUR2-rack",  "replaces": "Card 2000-139 or fibre optic to BUR2 rack",
             "confidence": 0.92},
    "500B": {"board_id": "BUR3_COMM",  "description": "BUR3 MVB communication / CHBA Interface",
             "part_ref": "BUR3-rack",  "replaces": "Card 2000-139 or fibre optic to BUR3 rack",
             "confidence": 0.92},
    "500D": {"board_id": "BUR2_COMM",  "description": "BUR2/B2AUXC2 MVB communication",
             "part_ref": "BUR2-rack",  "replaces": "Card 2000-139 or fibre optic to BUR2 rack",
             "confidence": 0.90},
    # Fire detection — ECode 5091, EnvBl EG_STB2_HBB2
    "5091": {"board_id": "SPIF_FIRE",  "description": "Fire detection input / STB2 cubicle",
             "part_ref": "pos.212",    "replaces": "Fire detection sensor module (pos. 212)",
             "confidence": 0.88},
    # VCB stuck — ECode 5071, EnvBl EG_STB1_HBB1
    "5071": {"board_id": "SPIF_VCB",   "description": "VCB holding circuit / SPIF output",
             "part_ref": "CON-A601",   "replaces": "Check VCB pos.5 before replacing A601",
             "confidence": 0.85},
    # MR pressure — ECode 5029, EnvBl EG_FLG1
    "5029": {"board_id": "SPIF_MR",    "description": "Main reservoir pressure input / SPIF",
             "part_ref": "CON-A601",   "replaces": "Check compressor MCBs before A601",
             "confidence": 0.82},
    # Coolant — ECode 5165 (CCUO-level), 3423/4423 (DCU sensor level)
    "5165": {"board_id": "COOLANT_SYS","description": "Coolant pump system / SPIF monitoring",
             "part_ref": "CON-A601",   "replaces": "Check coolant pump MCBs 63.1/1-63.1/2",
             "confidence": 0.85},
    "3423": {"board_id": "COOL_DCU1",  "description": "Coolant pressure sensor — DCU1 (Bogie 1)",
             "part_ref": "CON1-sensor","replaces": "Check coolant pressure sensor and MCB 63.1/1",
             "confidence": 0.90},
    "4423": {"board_id": "COOL_DCU2",  "description": "Coolant pressure sensor — DCU2 (Bogie 2)",
             "part_ref": "CON2-sensor","replaces": "Check coolant pressure sensor and MCB 63.1/2",
             "confidence": 0.90},
    "3425": {"board_id": "COOL_DCU1",  "description": "Coolant pressure below minimum — DCU1",
             "part_ref": "CON1-sensor","replaces": "Check coolant pump MCB 63.1/1 and expansion tank",
             "confidence": 0.90},
    "4425": {"board_id": "COOL_DCU2",  "description": "Coolant pressure below minimum — DCU2",
             "part_ref": "CON2-sensor","replaces": "Check coolant pump MCB 63.1/2 and expansion tank",
             "confidence": 0.90},
    # Precharge — ECode 5124/5125, EnvBl EG_VCI
    "5124": {"board_id": "PRECHARGE_1","description": "CON1 precharge circuit / VCI",
             "part_ref": "CON1-VCI",   "replaces": "Check precharge resistor and MCB CON1 side",
             "confidence": 0.85},
    "5125": {"board_id": "PRECHARGE_2","description": "CON2 precharge circuit / VCI",
             "part_ref": "CON2-VCI",   "replaces": "Check precharge resistor and MCB CON2 side",
             "confidence": 0.85},
    # Trafo sensors — ECode 5113/5114/513C/513D, EnvBl EG_VCI
    "5113": {"board_id": "TRAFO_SENS", "description": "Transformer temperature sensor 1 — VCI",
             "part_ref": "SLG-sensor", "replaces": "Check SLG1 temperature sensor wiring",
             "confidence": 0.88},
    "5114": {"board_id": "TRAFO_SENS", "description": "Transformer temperature sensor 2 — VCI",
             "part_ref": "SLG-sensor", "replaces": "Check SLG2 temperature sensor wiring",
             "confidence": 0.88},
    "513C": {"board_id": "TRAFO_SENS", "description": "Transformer temperature sensor 3 not valid",
             "part_ref": "SLG-sensor", "replaces": "Check SLG temp sensor wiring and VCI board",
             "confidence": 0.88},
    "513D": {"board_id": "TRAFO_SENS", "description": "Transformer temperature sensor 4 not valid",
             "part_ref": "SLG-sensor", "replaces": "Check SLG temp sensor wiring and VCI board",
             "confidence": 0.88},
    # IGBT feedback — ECode 3105 (DCU1/M1), EnvBl S_CvDgEnvGrM1_1
    "3105": {"board_id": "DCUM1_1",   "description": "Motor Converter 1 Control — DCU1 (IGBT feedback)",
             "part_ref": "CON1-A605-A02","replaces": "Card 1669 (Dual IGBT Gate Driver) on CON1-A102",
             "confidence": 0.95},
    # OHE/MVB cascade — ECode 3426/4426, EnvBl S_CvDgEnvGrG1_1/2
    "3426": {"board_id": "MVB_PULL_1", "description": "MVB pullwire from DCU2 — cascade to DCU1",
             "part_ref": "CON1-MVB",   "replaces": "Trace OHE fault origin — not a card replacement",
             "confidence": 0.88},
    "4426": {"board_id": "MVB_PULL_2", "description": "MVB pullwire from DCU1 — cascade to DCU2",
             "part_ref": "CON2-MVB",   "replaces": "Trace OHE fault origin — not a card replacement",
             "confidence": 0.88},
    # BUR internal faults — ECode 1xxx/2xxx, EnvBl EG_BUR1/2/3
    "10":   {"board_id": "BUR1_INT",   "description": "BUR1 internal fault (aux converter)",
             "part_ref": "EG_BUR1",    "replaces": "BUR1 rack inspection — Card 1302-1 or inverter",
             "confidence": 0.92},
    "15":   {"board_id": "BUR2_INT",   "description": "BUR2 internal fault (aux converter)",
             "part_ref": "EG_BUR2",    "replaces": "BUR2 rack inspection — Card 1302-1 or inverter",
             "confidence": 0.92},
    "20":   {"board_id": "BUR3_INT",   "description": "BUR3 internal fault (aux converter)",
             "part_ref": "EG_BUR3",    "replaces": "BUR3 rack inspection — Card 1302-1 or inverter",
             "confidence": 0.92},
}

# ---------------------------------------------------------------------------
# EVENT NAME SUFFIX → BOARD REFINEMENT
# ---------------------------------------------------------------------------
# Applied after the ECode prefix to confirm or override the board assignment.
# These are regex patterns matched against the full Event Name string.
# Higher confidence than ECode alone because Event Name is software-generated
# and directly references the card/slot that raised the flag.

EVENT_NAME_BOARD_PATTERNS: List[Tuple[str, dict]] = [
    # BUR MVB dropout — XU_EMVBDistBURn
    (r"XU_EMVBDistBUR1",   {"board_id": "BUR1_COMM","confidence_boost": 0.05,
                             "replaces": "Card 2000-139 or fibre optic to BUR1 rack"}),
    (r"XU_EMVBDistBUR2",   {"board_id": "BUR2_COMM","confidence_boost": 0.05,
                             "replaces": "Card 2000-139 or fibre optic to BUR2 rack"}),
    (r"XU_EMVBDistBUR3",   {"board_id": "BUR3_COMM","confidence_boost": 0.05,
                             "replaces": "Card 2000-139 or fibre optic to BUR3 rack"}),
    # BUR internal inverter / overcurrent faults
    (r"XU_EInv(erter)?Flt|EInvFlt", {"board_id": "BUR_INT","confidence_boost": 0.05,
                             "replaces": "BUR rack inverter — inspect Card 1302-1"}),
    (r"XU_EAuxConvOvTmp",  {"board_id": "BUR_INT", "confidence_boost": 0.05,
                             "replaces": "BUR thermal management — check airflow and sensors"}),
    # IGBT feedback failures — D_IGTFbFl or D_FpgaM pattern
    (r"D_IGTFbFl.*M1|D_FpgaM1", {"board_id": "DCUM1_1","confidence_boost": 0.05,
                             "replaces": "Card 1669 (Dual IGBT Gate Driver) — CON1-A102/M1"}),
    (r"D_IGTFbFl.*M2|D_FpgaM2", {"board_id": "DCUM2_1","confidence_boost": 0.05,
                             "replaces": "Card 1669 (Dual IGBT Gate Driver) — CON1-A102/M2"}),
    (r"D_IGTFbFl.*M3|D_FpgaM3", {"board_id": "DCUM3_1","confidence_boost": 0.05,
                             "replaces": "Card 1669 (Dual IGBT Gate Driver) — CON1-A102/M3"}),
    # VCB stuck — MC_EMCBStkOn
    (r"MC_EMCBStkOn",      {"board_id": "SPIF_VCB","confidence_boost": 0.05,
                             "replaces": "Check VCB pos.5 before replacing A601"}),
    # Fire detection — SF_EFlrFiDet
    (r"SF_EFlrFiDet",      {"board_id": "SPIF_FIRE","confidence_boost": 0.05,
                             "replaces": "Fire detection sensor module (pos. 212)"}),
    # MR pressure — XCV_EpMnResNOK
    (r"XCV_EpMnResNOK",    {"board_id": "SPIF_MR", "confidence_boost": 0.05,
                             "replaces": "Check compressor MCBs before A601"}),
    # Coolant pressure — D_XQCo.*SPIF
    (r"D_XQCo.*SPIF.*CON1", {"board_id": "COOL_DCU1","confidence_boost": 0.05,
                             "replaces": "Check coolant pressure sensor and MCB 63.1/1"}),
    (r"D_XQCo.*SPIF.*CON2", {"board_id": "COOL_DCU2","confidence_boost": 0.05,
                             "replaces": "Check coolant pressure sensor and MCB 63.1/2"}),
    (r"XO[12]_ECoolPmp",   {"board_id": "COOLANT_SYS","confidence_boost": 0.05,
                             "replaces": "Check coolant pump MCBs 63.1/1 and 63.1/2"}),
    # Precharge — AM[12]_EPreCgWaitRun
    (r"AM1_EPreCgWaitRun",  {"board_id": "PRECHARGE_1","confidence_boost": 0.05,
                             "replaces": "Check precharge resistor CON1 side"}),
    (r"AM2_EPreCgWaitRun",  {"board_id": "PRECHARGE_2","confidence_boost": 0.05,
                             "replaces": "Check precharge resistor CON2 side"}),
    # Trafo sensors — MT_ETfNNotValid
    (r"MT_ETf[1234]NotValid|MT_ETf[1234]", {"board_id": "TRAFO_SENS","confidence_boost": 0.05,
                             "replaces": "Check SLG temperature sensor wiring"}),
    # MVB pullwire / OHE cascade
    (r"D_MvbPwAvSpif.*CON1", {"board_id": "MVB_PULL_1","confidence_boost": 0.05,
                              "replaces": "Trace OHE fault on DCU1 side"}),
    (r"D_MvbPwAvSpif.*CON2", {"board_id": "MVB_PULL_2","confidence_boost": 0.05,
                              "replaces": "Trace OHE fault on DCU2 side"}),
    # Panto pressure — MPV_EPgXNoPres / MPV_EPgDown
    (r"MPV_EPg.*NoPres|MPV_EPgDown", {"board_id": "SPIF_PAN","confidence_boost": 0.05,
                             "replaces": "Check pantograph pressure supply and MCB 80.1"}),
]

# ---------------------------------------------------------------------------
# ENVBL ID# ---------------------------------------------------------------------------
# ENVBL ID → PROCESSOR ORIGIN MAP
# ---------------------------------------------------------------------------
# Maps EnvBl Id prefixes to originating processor / side.
# When EnvBl disagrees with ECode prefix → cross-DCU cascade flag.

ENVBL_PROCESSOR_MAP: Dict[str, str] = {
    # Confirmed from actual DDS export EnvBl Id column values
    "EG_FLG":           "FLG (Vehicle Control Unit / CCUO)",
    "EG_FLG1":          "FLG1 (Vehicle Control Unit — Cab 1)",
    "EG_FLG2":          "FLG2 (Vehicle Control Unit — Cab 2)",
    "EG_STB1_HBB1":     "STB1/HBB1 (Cubicle Control / Cab 1 side)",
    "EG_STB2_HBB2":     "STB2/HBB2 (Cubicle Control / Cab 2 side)",
    "EG_VCI":           "VCI (Vehicle Converter Interface / SPIF area)",
    "EG_BUR1":          "BUR1 (Aux Converter 1 — internal fault)",
    "EG_BUR2":          "BUR2 (Aux Converter 2 — internal fault)",
    "EG_BUR3":          "BUR3 (Aux Converter 3 — internal fault)",
    # S_CvDgEnvGr format: subsystem + DCU side
    # Pattern: S_CvDgEnvGr{subsys}_{side}  where side 1=DCU1/Bogie1, 2=DCU2/Bogie2
    "S_CvDgEnvGrL_1":   "DCU1 Line Converter (Bogie 1)",
    "S_CvDgEnvGrL_2":   "DCU2 Line Converter (Bogie 2)",
    "S_CvDgEnvGrM1_1":  "DCU1 Motor Converter 1 (Bogie 1)",
    "S_CvDgEnvGrM1_2":  "DCU2 Motor Converter 1 (Bogie 2)",
    "S_CvDgEnvGrM2_1":  "DCU1 Motor Converter 2 (Bogie 1)",
    "S_CvDgEnvGrM2_2":  "DCU2 Motor Converter 2 (Bogie 2)",
    "S_CvDgEnvGrM3_1":  "DCU1 Motor Converter 3 (Bogie 1)",
    "S_CvDgEnvGrM3_2":  "DCU2 Motor Converter 3 (Bogie 2)",
    "S_CvDgEnvGrG1_1":  "DCU1 Gate/SPIF area (Bogie 1)",
    "S_CvDgEnvGrG1_2":  "DCU2 Gate/SPIF area (Bogie 2)",
    "S_CvDgEnvGrG2_1":  "DCU1 Coolant/Thermal sensors (Bogie 1)",
    "S_CvDgEnvGrG2_2":  "DCU2 Coolant/Thermal sensors (Bogie 2)",
}

# EnvBl → DCU side and subsystem type for cascade detection
# Returns (dcu_side: int, subsystem: str) or None
def parse_envbl(envbl: str):
    """Parse S_CvDgEnvGr format: returns (side_int, subsys_str) or None."""
    import re as _re
    m = _re.match(r"S_CvDgEnvGr([A-Z0-9]+)_([12])", envbl)
    if m:
        return int(m.group(2)), m.group(1)
    return None


# ---------------------------------------------------------------------------
# DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class BoardLocalisation:
    """Result of localising a single event row or chain to a physical board."""
    board_id:    str                   # internal board identifier
    description: str                   # human-readable board name
    part_ref:    str                   # Bombardier part reference
    replaces:    str                   # what to physically swap
    confidence:  float                 # 0.0–1.0
    source:      str                   # how localisation was determined
    # Optional extra context
    note:        Optional[str] = None  # additional note (e.g., cascade)
    processor_origin: Optional[str] = None   # from EnvBl Id
    cascade_suspected: bool = False    # True when EnvBl disagrees with ECode side
    # Which columns contributed
    ecode_used:      bool = False
    event_name_used: bool = False
    envbl_used:      bool = False


def _unknown_board() -> BoardLocalisation:
    return BoardLocalisation(
        board_id    = "UNKNOWN",
        description = "Board not identifiable from available columns",
        part_ref    = "—",
        replaces    = "Physical inspection required",
        confidence  = 0.0,
        source      = "none",
    )


# ---------------------------------------------------------------------------
# SINGLE-ROW LOCALISER
# ---------------------------------------------------------------------------

def localise_board(row: pd.Series) -> BoardLocalisation:
    """
    Attempt to localise a single DDS event row to a physical board.

    Priority order:
    1. Exact ECode 0 match (4-char hex — most specific)
    2. ECode 0 prefix match (2-char hex)
    3. Event Name pattern match (refines or overrides ECode result)
    4. EnvBl Id for cross-DCU cascade detection

    Returns a BoardLocalisation dataclass.
    """
    ecode      = str(row.get("ECode 0",   "")).strip().upper()
    event_name = str(row.get("Event Name","")).strip()
    envbl      = str(row.get("EnvBl Id",  "")).strip()
    dist_text  = str(row.get("Dist Text", "")).strip()

    result = _unknown_board()

    # ── Step 1: ECode exact match (4-char) ──
    if len(ecode) >= 4:
        key4 = ecode[:4]
        if key4 in ECODE_BOARD_MAP:
            m = ECODE_BOARD_MAP[key4]
            result = BoardLocalisation(
                board_id    = m["board_id"],
                description = m["description"],
                part_ref    = m["part_ref"],
                replaces    = m["replaces"],
                confidence  = m["confidence"],
                source      = f"ECode exact ({key4})",
                ecode_used  = True,
            )

    # ── Step 2: ECode prefix match (2-char) if no exact match ──
    if result.board_id == "UNKNOWN" and len(ecode) >= 2:
        key2 = ecode[:2]
        if key2 in ECODE_BOARD_MAP:
            m = ECODE_BOARD_MAP[key2]
            result = BoardLocalisation(
                board_id    = m["board_id"],
                description = m["description"],
                part_ref    = m["part_ref"],
                replaces    = m["replaces"],
                confidence  = m["confidence"],
                source      = f"ECode prefix ({key2})",
                ecode_used  = True,
            )

    # ── Step 3: Event Name pattern matching ──
    if event_name and event_name not in ("nan", ""):
        for pattern, override in EVENT_NAME_BOARD_PATTERNS:
            if re.search(pattern, event_name, re.IGNORECASE):
                if "note" in override:
                    # Don't replace existing board — just add the note
                    result.note             = override.get("note")
                    result.event_name_used  = True
                    result.confidence       = min(
                        result.confidence + override.get("confidence_boost", 0), 1.0
                    )
                else:
                    # Upgrade or replace the board assignment
                    boost = override.get("confidence_boost", 0.0)
                    if result.board_id == "UNKNOWN":
                        # Fully override with Event Name result
                        result.board_id    = override.get("board_id", result.board_id)
                        result.replaces    = override.get("replaces",  result.replaces)
                        result.source      = f"Event Name pattern: {pattern}"
                        result.confidence  = 0.5 + boost
                    else:
                        # Confirm existing board — boost confidence
                        if override.get("board_id", result.board_id).startswith(result.board_id[:4]):
                            result.confidence = min(result.confidence + boost, 1.0)
                            result.source    += f" + Event Name ({pattern})"
                        else:
                            # Event Name suggests a different board — take the more specific
                            result.board_id  = override.get("board_id", result.board_id)
                            result.replaces  = override.get("replaces",  result.replaces)
                            result.confidence = min(result.confidence + boost, 1.0)
                            result.source    += f" + Event Name override ({pattern})"
                    result.event_name_used = True
                break   # first matching pattern wins

    # ── Step 4: EnvBl Id — processor origin and precise cascade detection ──
    if envbl and envbl not in ("nan", ""):
        # Direct lookup first
        for prefix, processor in ENVBL_PROCESSOR_MAP.items():
            if envbl == prefix or envbl.startswith(prefix + "_"):
                result.processor_origin = processor
                result.envbl_used       = True
                break
        if not result.processor_origin and envbl in ENVBL_PROCESSOR_MAP:
            result.processor_origin = ENVBL_PROCESSOR_MAP[envbl]
            result.envbl_used = True

        # Parse S_CvDgEnvGr format for precise DCU side + subsystem
        parsed = parse_envbl(envbl)
        if parsed:
            envbl_side, envbl_subsys = parsed
            result.envbl_used = True
            # ECode prefix encodes DCU side: 3x = DCU1, 4x = DCU2
            ecode_side = (1 if ecode[:1] == "3" else
                          2 if ecode[:1] == "4" else 0)
            if ecode_side and ecode_side != envbl_side:
                result.cascade_suspected = True
                origin_side  = f"DCU{ecode_side} (Bogie {ecode_side})"
                cascade_side = f"DCU{envbl_side} (Bogie {envbl_side})"
                result.note = (
                    (result.note or "") +
                    f" [CROSS-DCU CASCADE: fault originated in {origin_side}"
                    f" but reported via {cascade_side} EnvBl — "
                    f"investigate {origin_side} converter]"
                )
                # Confidence reduction on cascade — reporting side != fault side
                result.confidence = max(result.confidence - 0.1, 0.5)
            elif ecode_side and ecode_side == envbl_side:
                # EnvBl confirms the ECode side — small confidence boost
                result.confidence = min(result.confidence + 0.03, 1.0)
                result.source += f" + EnvBl confirmed (side {envbl_side})"
        elif result.envbl_used:
            # Named EnvBl (EG_BUR1 etc.) — check for BUR-originated faults
            if "BUR1" in envbl and result.board_id not in ("BUR1_COMM","BUR1_INT"):
                # Event came from BUR1 internal — update board if still generic SPIF
                if result.board_id.startswith("SPIF"):
                    result.board_id = "BUR1_INT"
                    result.description = "BUR1 internal fault"
                    result.replaces = "BUR1 rack inspection — Card 1302-1 or inverter"
                    result.confidence = min(result.confidence + 0.1, 1.0)
                    result.source += " + EnvBl BUR1 override"
            elif "BUR2" in envbl and result.board_id not in ("BUR2_COMM","BUR2_INT"):
                if result.board_id.startswith("SPIF"):
                    result.board_id = "BUR2_INT"
                    result.description = "BUR2 internal fault"
                    result.replaces = "BUR2 rack inspection — Card 1302-1 or inverter"
                    result.confidence = min(result.confidence + 0.1, 1.0)
                    result.source += " + EnvBl BUR2 override"
            elif "BUR3" in envbl and result.board_id not in ("BUR3_INT",):
                if result.board_id.startswith("SPIF"):
                    result.board_id = "BUR3_INT"
                    result.description = "BUR3 internal fault"
                    result.replaces = "BUR3 rack inspection — Card 1302-1 or inverter"
                    result.confidence = min(result.confidence + 0.1, 1.0)
                    result.source += " + EnvBl BUR3 override"

    return result


# ---------------------------------------------------------------------------
# PCB SUSPECT SCORER
# ---------------------------------------------------------------------------
# This is the layer that identifies probable card-level faults even when
# no single event is conclusive — it looks at the PATTERN of events across
# a session or chain and scores behavioural indicators.

# Weighted signals that suggest a PCB/card fault rather than a system fault
PCB_SUSPECT_SIGNALS = {
    # Communication loss to a specific DCU2 board — strong card indicator
    "communication loss to DCUL":   8,
    "communication loss to DCUM1":  8,
    "communication loss to DCUM2":  8,
    "communication loss to DCUM3":  8,
    "communication loss to SPIF":   8,
    # Lifesign loss — MVB communication failure, usually fibre optic or board
    "Lifesign from":                5,
    # IGBT feedback — gate driver card (Card 1669)
    "IGBT feedback":               10,
    "IGBT4 feedback":              10,
    # FPGA protective shutdown — often consequence, but repeated = board issue
    "FPGA caused PS":               4,
    "D_PrSdByFpga":                 4,
    # Feedback fault on contactor — either the contactor or the digital input
    "feedback fault":               6,
    "not closed":                   3,
    "not opened":                   3,
    # PSU implausible — power supply board
    "PSU for DCU":                  7,
    "implausible":                  2,   # generic plausibility flag
    # DC link voltage sensor — measurement chain issue
    "volt.sensor implaus":          6,
    "UDC2-UDC3 implausible":        7,
    # Speed sensor — may be sensor wiring but also DCU2/M input failure
    "SS no signal":                 5,
    "SS no supply":                 6,
    # Motor temperature sensor
    "no Temp.":                     4,
    "Temp. implaus.":               5,
    # Card cage fan — indicator of Card Cage assembly issue
    "fan faulty":                   6,
    # Self-test failure — always a board-level finding
    "Self-test periph. I/O failed": 9,
    # HW jumper — human error or new board fitted incorrectly
    "HW Jumper problem":            3,
    # MOBAD battery — SPIF board internal battery
    "low battery on Mobad":         5,
}

# Score thresholds
PCB_SCORE_THRESHOLDS = {
    "CONFIRMED_CARD":   25,   # Replace the card
    "PROBABLE_CARD":    15,   # Inspect card; replace if visual inspection confirms
    "POSSIBLE_CARD":     8,   # Check connections first; card may be secondary
    "SYSTEM_FAULT":      0,   # Below 8: more likely a system/wiring/sensor issue
}


@dataclass
class PCBSuspectResult:
    score:         int
    verdict:       str            # one of PCB_SCORE_THRESHOLDS keys
    contributing:  List[Tuple[str, int]]   # [(signal, score_contribution), ...]
    top_board:     Optional[BoardLocalisation] = None   # highest-confidence board from events
    recommendation: str = ""


def score_pcb_suspect(
    events_df: pd.DataFrame,
    context: str = "session"
) -> PCBSuspectResult:
    """
    Score a set of events (session or chain) for likelihood of a PCB/card fault.

    events_df : DataFrame of DDS events (Dist Text, Event Name, ECode 0, EnvBl Id)
    context   : "session" or "chain" — used in recommendation text only

    Returns a PCBSuspectResult.
    """
    total_score   = 0
    contributing  = []
    board_hits: List[BoardLocalisation] = []

    texts = events_df["Dist Text"].fillna("").tolist()
    event_names = events_df.get("Event Name", pd.Series(dtype=str)).fillna("").tolist()

    for signal, weight in PCB_SUSPECT_SIGNALS.items():
        # Check Dist Text
        count = sum(1 for t in texts if signal.lower() in str(t).lower())
        # Check Event Name for the same signal
        count += sum(1 for e in event_names if signal.lower() in str(e).lower())
        if count > 0:
            # Repeated occurrences add diminishing returns (log scaling)
            import math
            contribution = weight * (1 + 0.3 * math.log(count))
            contribution = round(contribution)
            total_score += contribution
            contributing.append((signal, contribution))

    # Localise each row and collect board votes
    for _, row in events_df.iterrows():
        bl = localise_board(row)
        if bl.board_id != "UNKNOWN" and bl.confidence >= 0.6:
            board_hits.append(bl)

    # Find the most-voted board
    top_board = None
    if board_hits:
        board_votes: Dict[str, List[BoardLocalisation]] = defaultdict(list)
        for b in board_hits:
            board_votes[b.board_id].append(b)
        top_id = max(board_votes, key=lambda k: (len(board_votes[k]),
                                                   max(b.confidence for b in board_votes[k])))
        # Take the highest-confidence instance of the top board
        top_board = max(board_votes[top_id], key=lambda b: b.confidence)

    # Determine verdict
    verdict = "SYSTEM_FAULT"
    for label, threshold in sorted(
        PCB_SCORE_THRESHOLDS.items(), key=lambda x: -x[1]
    ):
        if total_score >= threshold:
            verdict = label
            break

    # Build recommendation
    if verdict == "CONFIRMED_CARD" and top_board:
        rec = (
            f"PCB/card fault confirmed (score={total_score}). "
            f"Replace {top_board.replaces}. "
            f"Board: {top_board.description}. "
            f"Ref: {top_board.part_ref}."
        )
        if top_board.note:
            rec += f" Note: {top_board.note}"
    elif verdict == "PROBABLE_CARD" and top_board:
        rec = (
            f"Probable PCB/card fault (score={total_score}). "
            f"Inspect {top_board.description} ({top_board.part_ref}) visually. "
            f"Check connector seating and fibre optic cable integrity first. "
            f"Replace {top_board.replaces} if visual inspection is inconclusive."
        )
    elif verdict == "POSSIBLE_CARD" and top_board:
        rec = (
            f"Possible card involvement (score={total_score}). "
            f"Check wiring and connectors to {top_board.description} first. "
            f"Card replacement ({top_board.replaces}) only if wiring checks clean."
        )
    else:
        rec = (
            f"More likely a system/sensor/wiring fault (score={total_score}). "
            f"Check physical connections, sensor readings, and MCBs before "
            f"suspecting a card-level fault."
        )

    return PCBSuspectResult(
        score         = total_score,
        verdict       = verdict,
        contributing  = sorted(contributing, key=lambda x: -x[1]),
        top_board     = top_board,
        recommendation = rec,
    )


# ---------------------------------------------------------------------------
# CHAIN ANNOTATOR
# ---------------------------------------------------------------------------
# Called from dds_cross_session.process_file() to enrich chain results
# with board localisation and PCB suspect scores.

# ---------------------------------------------------------------------------
# CONFIDENCE-GRADUATED GUIDANCE
# ---------------------------------------------------------------------------
# The key principle: false precision is worse than honest uncertainty.
# A maintenance engineer who replaces the wrong specific card wastes time
# and money and may introduce new faults. A broader but correct direction
# (check this rack) is more useful than a confidently wrong card number.
#
# Guidance levels:
#   conf >= 0.85 + ATIL-corroborated card: name the specific card
#   conf >= 0.70: name the board/rack, suggest card family
#   conf >= 0.50: name the DCU side, suggest physical inspection
#   conf <  0.50: direction only, physical inspection required

def graduated_guidance(bl: BoardLocalisation, chain_id: str = "") -> str:
    """
    Return human-readable guidance at the appropriate precision level
    for the given board localisation confidence.
    """
    # ATIL-corroborated card mappings — only name a specific card when
    # fleet history confirms the card-to-fault relationship
    ATIL_CARD_MAP = {
        # Converter / IGBT cards
        "IGBT_FEEDBACK":        "Card 1669 (Dual IGBT Gate Driver) — check gate driver resistor paths on CON1-A102",
        "VCESAT_IGBT":          "Card 1669 (Dual IGBT Gate Driver) — Vce-sat protection firing; check gate driver card first",
        "CAN_COMM_BUR":         "Card 2000-138 (SCR/Converter Control) — check CAN bus termination and fibre optic links",
        "DC_LINK_OV_THYRISTOR": "Card 1703 (Single Thyristor Driver) — check thyristor gate circuit and snubber",
        # BUR / auxiliary converter cards
        "BUR_COOLANT_INTERFACE":"Card 1302-1 (CHBA Interface Card) — check in BUR rack (SB-1 for BUR-1, SB-2 for BUR-2/3)",
        "BUR_LIFESIGN_LOSS":    "Card 2000-139 (BUR Communication Card) or fibre optic cable to BUR rack — check fibre first",
        # System-level faults — no card replacement, physical checks only
        "COOLANT_FPGA":         "Check coolant pump MCB (63.1/1 HB1 or 63.1/2 HB2) and expansion tank level first — FPGA shutdown is consequence, not cause",
        "TRAFO_OIL":            "Check transformer oil pump MCBs 62.1/1 and 62.1/2 — this is not a card fault unless both MCBs healthy",
        "MR_PRESSURE_BRAKE":    "Check compressor MCBs 47.1/1 (HB1) and 47.1/2 (HB2), auto drain valves, and air dryer — not a SPIF card fault",
        "FIRE_DETECT_PERSISTENT":"Replace fire detection sensor module (pos. 212) if persistent >7 days — safety critical, do not defer",
        "PANTO_MCE_OFF":        "Check pantograph pressure supply and MCB 80.1 before suspecting SPIF — pneumatic fault is primary",
        "OHE_MVB_CASCADE":      "OHE-induced cascade — check DC link voltage recovery and MVB communication after OHE stabilises; no card replacement unless cascade persists after OHE clears",
        "VCB_STUCK_ON":         "Check VCB (pos. 5) armature, aux contacts, and pneumatic supply to trip coil — mechanical fault; check SPIF output only if VCB physically healthy",
        "PRECHARGE_OVERHEAT":   "Check precharge resistor condition and cooling airflow to converter — thermal fault; inspect Card 2000-138 area if resistor is healthy",
        # New BUR internal hardware faults
        "BUR3_NO_OUTPUT":       "BUR3 inverter stage hardware — check gate driver cards in BUR3 rack (SB-2); MCB 127.22/3",
        "BUR2_INVERTER_FAULT":  "BUR2 inverter power stage — check gate driver cards in BUR2 rack (SB-2); MCB 127.22/2",
    }

    if bl.board_id == "UNKNOWN" or bl.confidence < 0.3:
        return "Physical inspection required — board not identifiable from log data alone"

    board_name = bl.description
    # Only add DCU side qualifier when board_id actually encodes a side
    if "_1" in bl.board_id:
        dcu_side = " (DCU1 / Bogie 1)"
    elif "_2" in bl.board_id:
        dcu_side = " (DCU2 / Bogie 2)"
    else:
        dcu_side = ""

    atil_card = ATIL_CARD_MAP.get(chain_id, "")

    # Chains whose ATIL guidance is a system/physical action (not card replacement)
    SYSTEM_LEVEL_CHAINS = {
        "MR_PRESSURE_BRAKE", "TRAFO_OIL", "COOLANT_FPGA",
        "FIRE_DETECT_PERSISTENT", "PANTO_MCE_OFF", "OHE_MVB_CASCADE",
        "VCB_STUCK_ON", "PRECHARGE_OVERHEAT",
    }

    # chain_id is high-confidence evidence on its own — trust ATIL_CARD_MAP at >= 0.70
    if bl.confidence >= 0.70 and atil_card:
        if chain_id in SYSTEM_LEVEL_CHAINS:
            # No "inspect card cage" suffix — the guidance IS the action
            return f"{atil_card}"
        return (
            f"{board_name}{dcu_side}. "
            f"ATIL fleet history indicates: {atil_card}. "
            f"Inspect at {bl.replaces} before replacing."
        )
    elif bl.confidence >= 0.70:
        return (
            f"{board_name}{dcu_side}. "
            f"Inspect card cage ({bl.part_ref}). "
            f"Check all cards in this rack visually before replacing any single card."
        )
    elif bl.confidence >= 0.50:
        return (
            f"Likely in {board_name}{dcu_side} area. "
            f"Physical inspection of {bl.part_ref} rack required — "
            f"cannot narrow further from log data alone."
        )
    else:
        return (
            f"Fault in traction converter{dcu_side} area. "
            f"Physical inspection required — log data insufficient for card-level identification."
        )


def annotate_chains(results, df_full):
    """
    For each chain result, locate telemetry events using a +-5 min time window
    around trigger_time, run board localisation on the trigger row specifically,
    and return graduated guidance via ATIL_CARD_MAP.

    Returns dict: chain _uid -> {trigger_board, graduated_guidance, pcb_suspect}
    """
    annotations = {}

    for r in results:
        trigger_time  = getattr(r, "trigger_time",  None)
        trigger_text  = getattr(r, "trigger_text",  "") or ""
        chain_id      = getattr(r, "chain_id",      "") or ""
        terminal_time = getattr(r, "terminal_time", None)

        if trigger_time is None:
            annotations[r._uid] = {
                "trigger_board":      None,
                "graduated_guidance": "No trigger time — cannot localise.",
                "pcb_suspect":        None,
            }
            continue

        # ── Time window ──
        t_start = trigger_time - pd.Timedelta(minutes=5)
        t_end   = (terminal_time + pd.Timedelta(minutes=5)
                   if terminal_time else trigger_time + pd.Timedelta(hours=8))

        chain_events = df_full[
            (df_full["Start time"] >= t_start) &
            (df_full["Start time"] <= t_end)
        ].copy()

        if chain_events.empty:
            chain_events = df_full[
                (df_full["Start time"] >= trigger_time - pd.Timedelta(minutes=30)) &
                (df_full["Start time"] <= trigger_time + pd.Timedelta(minutes=30))
            ].copy()

        # ── Find trigger row by text match ──
        trigger_board = None
        trigger_rows  = pd.DataFrame()

        if not chain_events.empty and trigger_text:
            search_str = trigger_text[:40]
            trigger_rows = chain_events[
                chain_events["Dist Text"].str.contains(
                    re.escape(search_str), na=False, case=False, regex=True
                )
            ]

        if not trigger_rows.empty:
            trigger_board = localise_board(trigger_rows.iloc[0])
        elif not chain_events.empty:
            closest_idx = (chain_events["Start time"] - trigger_time).abs().idxmin()
            trigger_board = localise_board(chain_events.loc[closest_idx])

        # ── PCB scoring filtered to originating DCU side ──
        if (trigger_board and trigger_board.board_id != "UNKNOWN"
                and not trigger_rows.empty):
            t_ecode    = str(trigger_rows.iloc[0].get("ECode 0", "")).strip()
            dcu_prefix = t_ecode[0] if t_ecode else ""
            if dcu_prefix in ("3", "4"):
                side_events = chain_events[
                    chain_events["ECode 0"].apply(
                        lambda x: str(x).strip().startswith(dcu_prefix)
                        if pd.notna(x) else False
                    )
                ]
                score_events = side_events if len(side_events) >= 3 else chain_events
            else:
                score_events = chain_events
        else:
            score_events = chain_events

        pcb = score_pcb_suspect(score_events, context="chain") if not score_events.empty else None

        guidance = graduated_guidance(trigger_board, chain_id) if trigger_board else (
            "No board identified — physical inspection required."
        )

        annotations[r._uid] = {
            "trigger_board":      trigger_board,
            "graduated_guidance": guidance,
            "pcb_suspect":        pcb,
        }

    return annotations

def process_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n{'='*70}")
    print(f"  DDS Board Localiser — {path.name}")
    print(f"{'='*70}")

    df = pd.read_excel(file_path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["Start time"] = pd.to_datetime(df["Start time"], errors="coerce")
    df = df.dropna(subset=["Start time"]).sort_values("Start time").reset_index(drop=True)

    print(f"  Events : {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Show column availability
    has_ecode  = "ECode 0"    in df.columns
    has_evname = "Event Name" in df.columns
    has_envbl  = "EnvBl Id"   in df.columns
    print(f"  ECode 0: {'present' if has_ecode else 'MISSING'}  |  "
          f"Event Name: {'present' if has_evname else 'MISSING'}  |  "
          f"EnvBl Id: {'present' if has_envbl else 'MISSING'}")

    # Localise every event row
    localisations = df.apply(localise_board, axis=1)

    # Summary: board hit rate
    known     = [l for l in localisations if l.board_id != "UNKNOWN"]
    confident = [l for l in known if l.confidence >= 0.75]
    cascade   = [l for l in known if l.cascade_suspected]

    print(f"\n  Board localisation results:")
    print(f"    Total events    : {len(df)}")
    print(f"    Board identified: {len(known)} ({100*len(known)//len(df)}%)")
    print(f"    High confidence : {len(confident)} (conf ≥ 0.75)")
    print(f"    Cascade flagged : {len(cascade)}")

    # Board vote tally
    board_counts: Dict[str, int] = defaultdict(int)
    board_conf:   Dict[str, float] = defaultdict(float)
    for l in known:
        board_counts[l.board_id] += 1
        board_conf[l.board_id]    = max(board_conf[l.board_id], l.confidence)

    if board_counts:
        print(f"\n  Board vote tally (top 8):")
        for board_id, count in sorted(board_counts.items(), key=lambda x: -x[1])[:8]:
            sample = next(l for l in known if l.board_id == board_id)
            print(f"    {board_id:15s}  {count:4d} events  "
                  f"max_conf={board_conf[board_id]:.2f}  "
                  f"→ {sample.replaces[:55]}")

    # PCB suspect score on full file
    print(f"\n  PCB suspect score (full file):")
    pcb = score_pcb_suspect(df, context="file")
    print(f"    Score  : {pcb.score}")
    print(f"    Verdict: {pcb.verdict}")
    if pcb.top_board:
        print(f"    Top board: {pcb.top_board.description}")
        print(f"    Replacement: {pcb.top_board.replaces}")
    print(f"    Recommendation: {pcb.recommendation}")
    if pcb.contributing:
        print(f"    Contributing signals (top 5):")
        for sig, contrib in pcb.contributing[:5]:
            print(f"      +{contrib:3d}  {sig}")

    # Cross-DCU cascades
    if cascade:
        print(f"\n  Cross-DCU cascade events ({len(cascade)}):")
        for i, l in enumerate(cascade[:10]):
            row = df.iloc[i]  # approximate — just for display
            print(f"    {l.board_id}  conf={l.confidence:.2f}  {l.note[:80] if l.note else ''}")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dds_board_localiser.py <dds_excel_file>")
        print("\nAs a module, import annotate_chains() and call it after the")
        print("cross-session tracker runs — it enriches LinkedChainResult objects")
        print("with board localisation and PCB suspect scores.")
        sys.exit(0)

    for f in sys.argv[1:]:
        process_file(f)