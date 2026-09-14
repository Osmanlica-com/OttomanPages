"""
Model information and benchmarking utility.

Extracts, when available:
- model file size
- parameter count
- inference time (mean/std over repeated runs)
- input resolution
- hardware
- random seed
- confidence intervals (95%) and standard deviation across runs

Supported model formats:
- Ultralytics YOLO: .pt
- Detectron2/PyTorch: .pth
- PaddlePaddle: .pdparams

Usage:
    python info.py

Optional:
    python info.py --runs 5 --warmup 10 --size 1024
    python info.py --runs 10 --warmup 20 --size 640
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

MODEL_PATHS = [
    "/Users/ishak/Projects/Layout/runs/yolo11_seg_train4/weights/yolo11_seg_train4_best.pt",
    "/Users/ishak/Projects/Layout/runs/detect/train2/weights/best.pt",
    "/Users/ishak/Projects/Layout/runs/segment/train4/weights/best.pt",
    "/Users/ishak/Projects/Layout/Detectron/output/model_final.pth",
    "/Users/ishak/Projects/Layout/PageSeg/paddle/best_model/model.pdparams",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set common random seeds for reproducible benchmarking."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import paddle
        paddle.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict[str, Any]:
    """
    Calculate mean, sample standard deviation and approximate 95% CI.

    For n < 2, std/CI are reported as None.
    Uses a normal approximation (1.96 * SE), which is appropriate for
    benchmarking with a reasonably large number of repeated runs.
    """
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = statistics.mean(values)

    if len(values) < 2:
        return {
            "n": len(values),
            "mean": mean,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    std = statistics.stdev(values)
    z = 1.96
    se = std / math.sqrt(len(values))

    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_low": mean - z * se,
        "ci95_high": mean + z * se,
    }


# ---------------------------------------------------------------------------
# General model information
# ---------------------------------------------------------------------------

def file_info(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        return {
            "exists": False,
            "path": path,
            "file_size_mb": None,
        }

    return {
        "exists": True,
        "path": str(p),
        "file_name": p.name,
        "file_size_bytes": p.stat().st_size,
        "file_size_mb": p.stat().st_size / (1024 ** 2),
    }


def hardware_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "device": "CPU",
        "gpu": None,
        "gpu_count": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "CUDA"
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "MPS"
            info["gpu"] = "Apple Metal (MPS)"
    except Exception:
        pass

    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            info["paddle_device"] = "CUDA"
        else:
            info["paddle_device"] = "CPU"
    except Exception:
        pass

    return info


def count_parameters_torch(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Ultralytics YOLO
# ---------------------------------------------------------------------------

def load_yolo(path: str, size: int, device: str = "") -> Dict[str, Any]:
    """
    Load and benchmark an Ultralytics YOLO model.

    The exact model architecture is obtained from the checkpoint itself.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return {
            "framework": "ultralytics",
            "error": f"Ultralytics is not installed: {exc}",
        }

    try:
        model = YOLO(path)

        # Underlying PyTorch model.
        torch_model = model.model

        try:
            params = count_parameters_torch(torch_model)
        except Exception:
            params = None

        # Ultralytics model metadata.
        task = getattr(model, "task", None)
        names = getattr(model, "names", None)

        # Warm-up.
        dummy = None
        try:
            # A black image is enough for timing the forward/inference path.
            import numpy as np
            dummy = np.zeros((size, size, 3), dtype=np.uint8)
        except ImportError:
            pass

        return {
            "framework": "Ultralytics",
            "task": task,
            "parameters": params,
            "parameters_million": (
                params / 1e6 if params is not None else None
            ),
            "input_resolution": f"{size}x{size}",
            "class_count": len(names) if names is not None else None,
            "model_object": model,
            "dummy": dummy,
            "device": device or "auto",
        }

    except Exception as exc:
        return {
            "framework": "Ultralytics",
            "error": repr(exc),
        }


def benchmark_yolo(
    model: Any,
    dummy: Any,
    runs: int,
    warmup: int,
    size: int,
    device: str = "",
) -> Dict[str, Any]:
    """Benchmark YOLO inference in milliseconds."""
    import numpy as np

    timings = []

    # Warm-up.
    for _ in range(warmup):
        model.predict(
            source=dummy,
            imgsz=size,
            device=device or None,
            verbose=False,
        )

    for _ in range(runs):
        start = time.perf_counter()

        model.predict(
            source=dummy,
            imgsz=size,
            device=device or None,
            verbose=False,
        )

        # Synchronize CUDA so timing includes GPU work.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)

    stats = mean_std_ci(timings)

    return {
        "runs": runs,
        "warmup": warmup,
        "times_ms": timings,
        "inference_mean_ms": stats["mean"],
        "inference_std_ms": stats["std"],
        "inference_ci95_ms": [
            stats["ci95_low"],
            stats["ci95_high"],
        ],
        "inference_fps": (
            1000.0 / stats["mean"]
            if stats["mean"] and stats["mean"] > 0
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Detectron2
# ---------------------------------------------------------------------------

def load_detectron2(path: str, size: int) -> Dict[str, Any]:
    """
    Load Detectron2 checkpoint and extract model parameter count.

    Detectron2 requires its original config to construct the architecture.
    Therefore, this function searches common config locations next to the
    checkpoint and reports a clear message if no config is found.

    A .pth checkpoint alone normally does NOT contain enough information
    to reconstruct the Detectron2 model architecture.
    """
    try:
        import torch
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
    except ImportError as exc:
        return {
            "framework": "Detectron2",
            "error": f"Detectron2/PyTorch import failed: {exc}",
        }

    p = Path(path)

    candidate_configs = [
        p.parent / "config.yaml",
        p.parent / "config.yml",
        p.parent / "cfg.yaml",
        p.parent / "cfg.yml",
        p.parent.parent / "config.yaml",
        p.parent.parent / "config.yml",
        p.parent.parent / "cfg.yaml",
        p.parent.parent / "cfg.yml",
    ]

    config_path = next((x for x in candidate_configs if x.exists()), None)

    if config_path is None:
        return {
            "framework": "Detectron2",
            "error": (
                "Detectron2 checkpoint found, but no config YAML was found. "
                "Pass the training config path with --detectron-config."
            ),
        }

    try:
        cfg = get_cfg()
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(p)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(cfg)
        DetectionCheckpointer(model).load(str(p))
        model.eval()

        params = count_parameters_torch(model)

        return {
            "framework": "Detectron2",
            "config": str(config_path),
            "parameters": params,
            "parameters_million": params / 1e6,
            "input_resolution": f"{size}x{size}",
            "model_object": model,
            "cfg": cfg,
        }

    except Exception as exc:
        return {
            "framework": "Detectron2",
            "config": str(config_path),
            "error": repr(exc),
        }


def benchmark_detectron2(
    model: Any,
    cfg: Any,
    runs: int,
    warmup: int,
    size: int,
) -> Dict[str, Any]:
    """Benchmark a Detectron2 model using a synthetic image."""
    import numpy as np
    import torch

    device = next(model.parameters()).device

    image = torch.zeros(
        (3, size, size),
        dtype=torch.float32,
        device=device,
    )

    inputs = [{
        "image": image,
        "height": size,
        "width": size,
    }]

    timings = []

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

    stats = mean_std_ci(timings)

    return {
        "runs": runs,
        "warmup": warmup,
        "times_ms": timings,
        "inference_mean_ms": stats["mean"],
        "inference_std_ms": stats["std"],
        "inference_ci95_ms": [
            stats["ci95_low"],
            stats["ci95_high"],
        ],
        "inference_fps": (
            1000.0 / stats["mean"]
            if stats["mean"] and stats["mean"] > 0
            else None
        ),
    }


# ---------------------------------------------------------------------------
# PaddlePaddle
# ---------------------------------------------------------------------------

def load_paddle(path: str, size: int) -> Dict[str, Any]:
    """
    Load a Paddle .pdparams file.

    Important:
    A .pdparams file generally contains parameters only, not the complete
    network definition. Therefore parameter count can be extracted directly,
    but model construction/inference requires the original Paddle model code.
    """
    try:
        import paddle
    except ImportError as exc:
        return {
            "framework": "PaddlePaddle",
            "error": f"PaddlePaddle is not installed: {exc}",
        }

    try:
        state_dict = paddle.load(path)

        total_params = 0
        tensor_count = 0

        for value in state_dict.values():
            try:
                n = int(value.size)
                total_params += n
                tensor_count += 1
            except Exception:
                pass

        return {
            "framework": "PaddlePaddle",
            "parameters": total_params,
            "parameters_million": total_params / 1e6,
            "tensor_count": tensor_count,
            "input_resolution": f"{size}x{size}",
            "state_dict": state_dict,
        }

    except Exception as exc:
        return {
            "framework": "PaddlePaddle",
            "error": repr(exc),
        }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def inspect_model(
    path: str,
    runs: int,
    warmup: int,
    size: int,
    seed: int,
    device: str = "",
    detectron_config: Optional[str] = None,
) -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "model": Path(path).name,
        "path": path,
        "file": file_info(path),
        "seed": seed,
        "input_resolution": f"{size}x{size}",
        "hardware": hardware_info(),
        "benchmark": None,
        "notes": [],
    }

    if not Path(path).exists():
        result["notes"].append("Model file does not exist on this machine.")
        return result

    suffix = Path(path).suffix.lower()

    # -------------------- Ultralytics --------------------
    if suffix == ".pt":
        info = load_yolo(path, size, device)

        # Detectron2 is also a PyTorch .pth rather than .pt in this setup.
        result.update({
            k: v for k, v in info.items()
            if k not in {"model_object", "dummy"}
        })

        if "error" not in info:
            try:
                benchmark = benchmark_yolo(
                    info["model_object"],
                    info["dummy"],
                    runs=runs,
                    warmup=warmup,
                    size=size,
                    device=device,
                )
                result["benchmark"] = benchmark
            except Exception as exc:
                result["notes"].append(
                    f"YOLO inference benchmark failed: {repr(exc)}"
                )

        return result

    # -------------------- Detectron2 --------------------
    if suffix == ".pth":
        # If an explicit config was supplied, use it.
        if detectron_config:
            try:
                from detectron2.config import get_cfg
                from detectron2.modeling import build_model
                from detectron2.checkpoint import DetectionCheckpointer
                import torch

                cfg = get_cfg()
                cfg.merge_from_file(detectron_config)
                cfg.MODEL.WEIGHTS = path
                cfg.MODEL.DEVICE = (
                    device if device else
                    ("cuda" if torch.cuda.is_available() else "cpu")
                )

                model = build_model(cfg)
                DetectionCheckpointer(model).load(path)
                model.eval()

                params = count_parameters_torch(model)

                result.update({
                    "framework": "Detectron2",
                    "config": detectron_config,
                    "parameters": params,
                    "parameters_million": params / 1e6,
                })

                result["benchmark"] = benchmark_detectron2(
                    model,
                    cfg,
                    runs=runs,
                    warmup=warmup,
                    size=size,
                )

            except Exception as exc:
                result["framework"] = "Detectron2"
                result["notes"].append(
                    f"Detectron2 loading/benchmark failed: {repr(exc)}"
                )

            return result

        info = load_detectron2(path, size)

        result.update({
            k: v for k, v in info.items()
            if k not in {"model_object", "cfg"}
        })

        if "error" not in info:
            try:
                result["benchmark"] = benchmark_detectron2(
                    info["model_object"],
                    info["cfg"],
                    runs=runs,
                    warmup=warmup,
                    size=size,
                )
            except Exception as exc:
                result["notes"].append(
                    f"Detectron2 inference benchmark failed: {repr(exc)}"
                )

        return result

    # -------------------- PaddlePaddle --------------------
    if suffix == ".pdparams":
        info = load_paddle(path, size)

        result.update({
            k: v for k, v in info.items()
            if k != "state_dict"
        })

        result["notes"].append(
            "Paddle .pdparams contains weights/state_dict. "
            "The original Paddle network definition is required for "
            "reliable inference-time benchmarking."
        )

        return result

    result["notes"].append(f"Unsupported model extension: {suffix}")
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 110)
    print("MODEL INFORMATION")
    print("=" * 110)

    header = (
        f"{'Model':35s} "
        f"{'Size(MB)':>10s} "
        f"{'Params(M)':>12s} "
        f"{'Input':>10s} "
        f"{'Mean(ms)':>12s} "
        f"{'Std(ms)':>12s} "
        f"{'CI95(ms)':>24s}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        size_mb = r.get("file", {}).get("file_size_mb")
        params = r.get("parameters_million")

        bench = r.get("benchmark") or {}
        mean_ms = bench.get("inference_mean_ms")
        std_ms = bench.get("inference_std_ms")
        ci = bench.get("inference_ci95_ms")

        ci_text = (
            f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            if ci and ci[0] is not None
            else "N/A"
        )

        print(
            f"{r.get('model', '')[:35]:35s} "
            f"{size_mb:10.2f} " if size_mb is not None else
            f"{r.get('model', '')[:35]:35s} {'N/A':>10s} ",
            end=""
        )

        print(
            f"{params:12.3f} " if params is not None else f"{'N/A':>12s} ",
            end=""
        )

        print(
            f"{r.get('input_resolution', 'N/A'):>10s} "
            f"{mean_ms:12.3f} " if mean_ms is not None else
            f"{r.get('input_resolution', 'N/A'):>10s} {'N/A':>12s} ",
            end=""
        )

        print(
            f"{std_ms:12.3f} " if std_ms is not None else f"{'N/A':>12s} ",
            end=""
        )

        print(f"{ci_text:>24s}")

    print("=" * 110)


def make_json_serializable(obj: Any) -> Any:
    """Convert common NumPy/Paddle/Torch scalar objects to JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]

    try:
        return obj.item()
    except Exception:
        return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract model information and benchmark inference."
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measured inference runs (default: 10).",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up runs (default: 10).",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help=(
            "Synthetic/input benchmark resolution. "
            "Use the same value for all models for a fair comparison. "
            "Default: 1024."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Ultralytics/Detectron2 device, e.g. cuda:0, cpu, mps.",
    )

    parser.add_argument(
        "--detectron-config",
        type=str,
        default=None,
        help="Detectron2 config YAML for model_final.pth.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model_info.json",
        help="Output JSON filename (default: model_info.json).",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Seed             : {args.seed}")
    print(f"Input resolution : {args.size}x{args.size}")
    print(f"Measured runs    : {args.runs}")
    print(f"Warm-up runs     : {args.warmup}")

    results = []

    for path in MODEL_PATHS:
        print(f"\nProcessing: {path}")

        result = inspect_model(
            path=path,
            runs=args.runs,
            warmup=args.warmup,
            size=args.size,
            seed=args.seed,
            device=args.device,
            detectron_config=args.detectron_config,
        )

        results.append(result)

        if result.get("notes"):
            for note in result["notes"]:
                print(f"  NOTE: {note}")

    print_summary(results)

    output = {
        "benchmark_settings": {
            "seed": args.seed,
            "runs": args.runs,
            "warmup": args.warmup,
            "input_resolution": f"{args.size}x{args.size}",
            "device_argument": args.device or "auto",
            "confidence_level": 0.95,
        },
        "models": make_json_serializable(results),
    }

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nJSON report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
