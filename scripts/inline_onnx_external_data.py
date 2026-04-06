"""
Merge an ONNX model that uses external data (*.onnx + *.onnx.data) into a single .onnx file.

Use when ONNX Runtime reports: cannot find ... "model.onnx.data"

Requires: pip install onnx

Example:
  python scripts/inline_onnx_external_data.py models/sanet_partB_best.onnx -o models/sanet_partB_best_inlined.onnx

Then set CROWD_MODEL_PATH to the *_inlined.onnx file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Inline external ONNX weights into one file.")
    p.add_argument("onnx_path", type=Path, help="Path to the .onnx file (and .onnx.data in the same folder)")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <name>_inlined.onnx beside input)",
    )
    args = p.parse_args()
    onnx_path = args.onnx_path.resolve()
    if not onnx_path.is_file():
        print(f"Not found: {onnx_path}", file=sys.stderr)
        return 1

    try:
        import onnx
    except ImportError:
        print("Install the onnx package: pip install onnx", file=sys.stderr)
        return 1

    # Loads tensors from sibling external files when present
    model = onnx.load(str(onnx_path), load_external_data=True)
    out = args.out
    if out is None:
        out = onnx_path.parent / f"{onnx_path.stem}_inlined.onnx"
    else:
        out = out.resolve()

    onnx.save(model, str(out))
    print(f"Wrote single-file ONNX: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
