"""
Extract page-layout complexity statistics (Table 3) for the
Ottoman Page Segmentation train/test splits.

Layout characteristics:
  - Text blocks per page
  - Text-block area / width / height
  - Block density (% of page area covered, summing polygon areas)
  - Nearest-neighbor distance (px, centroid-to-centroid)
  - Overlap ratio (% of total block area that participates in pairwise AABB overlap)
  - Skew angle (°, absolute tilt of the longest polygon edge vs. horizontal)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "OttomanPageSegmentation"


def load_split(data_root: Path, split: str) -> dict[str, Any]:
    path = data_root / split / f"layout_{split}_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace area for a closed polygon given as [[x, y], ...]."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
    """Area-weighted centroid; falls back to vertex mean for degenerate polygons."""
    if not points:
        return 0.0, 0.0
    if len(points) < 3:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    a *= 0.5
    if abs(a) < 1e-9:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (6.0 * a), cy / (6.0 * a)


def aabb(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def skew_angle_deg(points: list[list[float]]) -> float:
    """
    Absolute angle (degrees) between the longest polygon edge and the
    horizontal axis, folded into [0, 45] so 0° means axis-aligned.
    """
    if len(points) < 2:
        return 0.0
    best_len = -1.0
    best_angle = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            if angle > 90.0:
                angle = 180.0 - angle
            if angle > 45.0:
                angle = 90.0 - angle
            best_angle = angle
    return best_angle


def summarize(values: list[float | int], ndigits: int = 2) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), ndigits),
        "median": round(statistics.median(values), ndigits),
        "min": round(min(values), ndigits) if isinstance(min(values), float) else min(values),
        "max": round(max(values), ndigits) if isinstance(max(values), float) else max(values),
        "std": round(statistics.pstdev(values), ndigits) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def analyze_layout_complexity(data: dict[str, Any]) -> dict[str, Any]:
    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    blocks_per_page: list[int] = []
    block_areas: list[float] = []
    block_widths: list[float] = []
    block_heights: list[float] = []
    block_densities: list[float] = []
    nn_distances: list[float] = []
    overlap_ratios: list[float] = []
    skew_angles: list[float] = []

    for image in data["images"]:
        image_id = image["id"]
        width = float(image.get("width") or 0)
        height = float(image.get("height") or 0)
        page_area = width * height
        anns = anns_by_image.get(image_id, [])
        blocks_per_page.append(len(anns))

        boxes: list[tuple[float, float, float, float]] = []
        areas: list[float] = []
        centroids: list[tuple[float, float]] = []

        for ann in anns:
            pts = ann.get("bbox") or []
            if not pts:
                continue
            area = polygon_area(pts)
            x0, y0, x1, y1 = aabb(pts)
            w = x1 - x0
            h = y1 - y0
            block_areas.append(area)
            block_widths.append(w)
            block_heights.append(h)
            skew_angles.append(skew_angle_deg(pts))
            boxes.append((x0, y0, x1, y1))
            areas.append(area)
            centroids.append(polygon_centroid(pts))

        if page_area > 0 and areas:
            block_densities.append(100.0 * sum(areas) / page_area)

        n = len(centroids)
        if n >= 2:
            for i in range(n):
                cx, cy = centroids[i]
                best = float("inf")
                for j in range(n):
                    if i == j:
                        continue
                    dx = cx - centroids[j][0]
                    dy = cy - centroids[j][1]
                    dist = math.hypot(dx, dy)
                    if dist < best:
                        best = dist
                nn_distances.append(best)

            overlap_area = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    overlap_area += aabb_intersection_area(boxes[i], boxes[j])
            total_area = sum(areas)
            if total_area > 0:
                overlap_ratios.append(100.0 * overlap_area / total_area)
            else:
                overlap_ratios.append(0.0)
        elif n == 1:
            # Single block: no neighbor / no pairwise overlap.
            overlap_ratios.append(0.0)

    return {
        "num_pages": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "text_blocks_per_page": summarize(blocks_per_page),
        "text_block_area_px2": summarize(block_areas),
        "text_block_width_px": summarize(block_widths),
        "text_block_height_px": summarize(block_heights),
        "block_density_pct": summarize(block_densities),
        "nearest_neighbor_distance_px": summarize(nn_distances),
        "overlap_ratio_pct": summarize(overlap_ratios),
        "skew_angle_deg": summarize(skew_angles),
    }


def format_row(name: str, stats: dict[str, Any]) -> str:
    if stats.get("mean") is None:
        return f"| {name} | — | — | — | — | — |"
    return (
        f"| {name} "
        f"| {stats['mean']} "
        f"| {stats['median']} "
        f"| {stats['min']} "
        f"| {stats['max']} "
        f"| {stats['std']} |"
    )


def print_table(split: str, stats: dict[str, Any]) -> None:
    print()
    print(f"### {split} (n={stats['num_pages']} pages, {stats['num_annotations']} blocks)")
    print()
    print("| Layout Characteristic | Mean | Median | Min | Max | Std. |")
    print("|---|---:|---:|---:|---:|---:|")
    rows = [
        ("Text blocks per page", stats["text_blocks_per_page"]),
        ("Text-block area (px²)", stats["text_block_area_px2"]),
        ("Text-block width (px)", stats["text_block_width_px"]),
        ("Text-block height (px)", stats["text_block_height_px"]),
        ("Block density (%)", stats["block_density_pct"]),
        ("Nearest-neighbor distance (px)", stats["nearest_neighbor_distance_px"]),
        ("Overlap ratio (%)", stats["overlap_ratio_pct"]),
        ("Skew angle (°)", stats["skew_angle_deg"]),
    ]
    for name, row in rows:
        print(format_row(name, row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to OttomanPageSegmentation root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to analyze.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to write full stats as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    report: dict[str, Any] = {"data_root": str(data_root)}
    print("=" * 72)
    print("Table 3 — Page layout complexity statistics")
    print("=" * 72)

    for split in args.splits:
        data = load_split(data_root, split)
        stats = analyze_layout_complexity(data)
        report[split] = stats
        print_table(split, stats)

    print()
    print("Notes:")
    print("  - Area uses polygon shoelace; width/height use axis-aligned extents.")
    print("  - Block density = 100 * sum(block areas) / page area (may exceed 100% if overlap).")
    print("  - Nearest-neighbor = min centroid distance to another block on the same page.")
    print("  - Overlap ratio = 100 * sum(pairwise AABB intersections) / sum(block areas), per page.")
    print("  - Skew = absolute tilt of longest edge vs horizontal, folded into [0, 45].")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON report to {args.save_json}")


if __name__ == "__main__":
    main()
