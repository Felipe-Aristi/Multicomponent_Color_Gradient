from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


POST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POST_DIR.parent
CONSTANTS_FILE = PROJECT_ROOT / "constants.cuh"


@dataclass(frozen=True)
class CaseConfig:
    nx: int
    ny: int
    nz: int
    diameter: float
    jet_velocity: float
    re: int
    we: int
    noutput: int
    case_dir: Path
    mean_dir: Path
    profiles_dir: Path
    vti_dir: Path


def _parse_constant(text: str, name: str) -> float:
    pattern = rf"{name}\s*=\s*(?:static_cast<[^>]+>\()?([0-9.+\-eE]+)"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {name} in {CONSTANTS_FILE}")
    return float(match.group(1))


def load_case_config(case_dir: str | Path | None = None,
                     output_dir: str | Path | None = None) -> CaseConfig:
    text = CONSTANTS_FILE.read_text()

    nx = int(_parse_constant(text, "NX"))
    ny = int(_parse_constant(text, "NY"))
    nz = int(_parse_constant(text, "NZ"))
    jet_radius = _parse_constant(text, "jet_radius")
    jet_velocity = _parse_constant(text, "jet_velocity")
    re_value = int(round(_parse_constant(text, "Re")))
    we_value = int(round(_parse_constant(text, "We")))
    noutput = int(_parse_constant(text, "NOUTPUT"))

    resolved_case_dir = Path(case_dir) if case_dir is not None else (
        PROJECT_ROOT / "JET_VTK" / f"Re{re_value}_We{we_value}"
    )
    if not resolved_case_dir.is_absolute():
        resolved_case_dir = (PROJECT_ROOT / resolved_case_dir).resolve()

    profiles_dir = Path(output_dir) if output_dir is not None else (
        resolved_case_dir / "profiles"
    )
    if not profiles_dir.is_absolute():
        profiles_dir = (PROJECT_ROOT / profiles_dir).resolve()

    return CaseConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        diameter=2.0 * jet_radius,
        jet_velocity=jet_velocity,
        re=re_value,
        we=we_value,
        noutput=noutput,
        case_dir=resolved_case_dir,
        mean_dir=resolved_case_dir / "mean_profiles",
        profiles_dir=profiles_dir,
        vti_dir=resolved_case_dir / "vti_data",
    )


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-dir", default=None,
                        help="Case directory. Defaults to JET_VTK/Re<Re>_We<We> from constants.cuh.")
    parser.add_argument("--output-dir", default=None,
                        help="Profile output directory. Defaults to <case-dir>/profiles.")
