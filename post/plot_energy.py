from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from case_config import add_case_args, load_case_config
from exportAndCrop import export_and_crop
from prettyPlot import pretty_plot


BLUE = "#009DDC"


def total_kinetic_energy(field: str,
                         case_dir: Path | None = None,
                         output_dir: Path | None = None) -> None:
    cfg = load_case_config(case_dir, output_dir)
    cfg.profiles_dir.mkdir(parents=True, exist_ok=True)

    input_path = cfg.mean_dir / f"tke_{field}.bin"
    tke = np.fromfile(input_path, dtype=np.float32)
    if tke.size == 0:
        raise ValueError(f"No samples found in {input_path}")

    if field != "diff":
        values = tke / (9.0 * cfg.jet_velocity * cfg.jet_velocity)
        min_ylim = float(np.min(values) * 0.95)
        max_ylim = float(np.max(values) * 1.05)
        if abs(max_ylim - min_ylim) < 1.0e-20:
            max_ylim = min_ylim + 1.0e-12
    else:
        values = tke
        positive = values[values > 0.0]
        min_ylim = float(max(np.min(positive) * 0.5, 1.0e-10)) if positive.size else 1.0e-10
        max_ylim = float(max(np.max(values) * 2.0, 1.0e-8))

    t_star = np.arange(values.size, dtype=float) * cfg.noutput * cfg.jet_velocity / cfg.diameter

    paper_width = 612
    margin_points = 54

    fig, ax, _ = pretty_plot(
        xLim=(0, float(t_star[-1]) if t_star.size > 1 else 1.0),
        yLim=(min_ylim, max_ylim),
        cLim=(-1, 1),
        plotAspectRatio=(1, 1, 1),
        xLabel=r"$t^*$",
        yLabel=r"$E_K^*$" if field != "diff" else r"$\Delta E_K^*$",
        yScientificNotation=True,
        yTickFormat=2,
        nxTicks=5,
        nyTicks=9,
        useColorBar=False,
        paperPoints=paper_width,
        marginPoints=margin_points,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=0,
        useGrid=True,
        fontSize=14,
        dpi=300,
    )

    ax.yaxis.set_label_coords(-0.125, 0.5)
    ax.plot(t_star, values, ls="-", lw=3, color=BLUE, markersize=5, label=f"Re {cfg.re}, We {cfg.we}")
    if field == "diff":
        ax.set_yscale("log")

    ax.grid(True)
    ax.legend(loc="best", fontsize=12)

    output_path = cfg.profiles_dir / f"tke_{field}.png"
    export_and_crop(fig, str(output_path))
    plt.close(fig)
    print(f"Energy plot written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TKE diagnostic histories.")
    add_case_args(parser)
    parser.add_argument("field", choices=["total", "avg", "diff"])
    args = parser.parse_args()

    total_kinetic_energy(
        field=args.field,
        case_dir=Path(args.case_dir) if args.case_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
