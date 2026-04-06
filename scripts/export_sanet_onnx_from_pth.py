"""
Build a single-file SANet ONNX (no separate .onnx.data) from the Kaggle-style checkpoint.

Requires: torch, sanet_partB_best.pth with key ``state_dict`` (same as the notebook).

Example:
  python scripts/export_sanet_onnx_from_pth.py models/sanet_partB_best.pth -o models/sanet_partB_best.onnx

Then keep CROWD_MODEL_PATH=./models/sanet_partB_best.onnx — ORT will load without .data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pth_path",
        type=Path,
        help="sanet_partB_best.pth (state_dict checkpoint)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output .onnx (default: same dir, same name as .pth)",
    )
    args = ap.parse_args()
    pth_path = args.pth_path.resolve()
    if not pth_path.is_file():
        print(f"Not found: {pth_path}", file=sys.stderr)
        return 1

    out = args.out
    if out is None:
        out = pth_path.with_suffix(".onnx")
    else:
        out = out.resolve()

    import torch

    from app.services.detection_service import SANet, _strip_module_prefix

    model = SANet()
    ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = _strip_module_prefix(ckpt["state_dict"])
    elif "model_state" in ckpt:
        sd = _strip_module_prefix(ckpt["model_state"])
    else:
        sd = _strip_module_prefix(ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 384, 512)
    tmp = out.with_suffix(".tmp.onnx")
    export_kw = dict(
        opset_version=17,
        input_names=["input"],
        output_names=["density_map"],
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )
    try:
        torch.onnx.export(model, dummy, str(tmp), dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(tmp), **export_kw)
    tmp.replace(out)
    print(f"Wrote single-file ONNX: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
