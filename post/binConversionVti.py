from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from case_config import add_case_args, load_case_config


def write_uy_avg_vti(case_dir: Path | None = None,
                     output_dir: Path | None = None) -> Path:
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError("The vtk Python package is required to write uy_avg.vti") from exc

    cfg = load_case_config(case_dir, output_dir)
    cfg.profiles_dir.mkdir(parents=True, exist_ok=True)

    input_path = cfg.mean_dir / "uy_avg.bin"
    output_path = cfg.profiles_dir / "uy_avg.vti"

    data = np.fromfile(input_path, dtype=np.float32)
    expected = cfg.nx * cfg.ny * cfg.nz
    if data.size != expected:
        raise ValueError(f"{input_path} has {data.size} values, expected {expected}")

    vtk_array = numpy_support.numpy_to_vtk(
        num_array=np.ascontiguousarray(data),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName("uy_avg")

    image = vtk.vtkImageData()
    image.SetDimensions(cfg.nx, cfg.ny, cfg.nz)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {output_path}")

    print(f"uy_avg VTI written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert uy_avg.bin to VTI.")
    add_case_args(parser)
    args = parser.parse_args()

    write_uy_avg_vti(
        case_dir=Path(args.case_dir) if args.case_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
