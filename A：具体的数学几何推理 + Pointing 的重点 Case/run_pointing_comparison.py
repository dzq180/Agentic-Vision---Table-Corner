import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from google import genai
from google.genai import types


WORK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORK_DIR.parent
DATASET_DIR = PROJECT_ROOT / "数据集"
GROUND_TRUTH_PATH = PROJECT_ROOT / "corner_dataset.json"
RESULTS_DIR = WORK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_PROMPT = """You are given one image captured by an orthographic camera.
Task: locate the 4 tabletop corner points of the table.

Use visual judgment only. Do not use geometric equations, vector sums, parallelogram constraints, or derive a hidden corner from the other three corners.

Ordering:
- corner_0: top-left
- corner_1: top-right
- corner_2: bottom-right
- corner_3: bottom-left

Coordinates:
- normalized to 0-1000
- each point uses [y, x]

Return only a JSON array:
[{"point":[y,x],"label":"corner_0"},{"point":[y,x],"label":"corner_1"},{"point":[y,x],"label":"corner_2"},{"point":[y,x],"label":"corner_3"}]
"""

GEOMETRY_PROMPT = """You are given one image captured by an orthographic camera.
Task: locate the 4 tabletop corner points of the table.

Use explicit geometric reasoning:
- the tabletop is a flat plane
- under orthographic projection, the tabletop forms a parallelogram in the image
- the 4 corners are the 4 vertices of that parallelogram
- opposite sides are parallel, so corner_0 + corner_2 = corner_1 + corner_3
- if one corner is unclear, compute it from the other three instead of guessing
- the diagonals intersect at the center of the parallelogram

Ordering:
- corner_0: top-left
- corner_1: top-right
- corner_2: bottom-right
- corner_3: bottom-left

Coordinates:
- normalized to 0-1000
- each point uses [y, x]
- decimals are allowed

Return only a JSON array:
[{"point":[y,x],"label":"corner_0"},{"point":[y,x],"label":"corner_1"},{"point":[y,x],"label":"corner_2"},{"point":[y,x],"label":"corner_3"}]
"""


@dataclass
class Case:
    image_name: str
    gt_corners: list[list[float]]


def _relay_base_url() -> str | None:
    for key in ("GOOGLE_GEMINI_BASE_URL", "GEMINI_BASE_URL", "GOOGLE_GENAI_BASE_URL"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip().rstrip("/")
    return None


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError('Missing GEMINI_API_KEY. Example: $env:GEMINI_API_KEY="your-key"')
    base_url = _relay_base_url()
    if base_url:
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(base_url=base_url))
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def load_cases() -> list[Case]:
    data = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return [Case(image_name=item["image"], gt_corners=item["corners"]) for item in data]


def clamp_point(y: float, x: float) -> list[float]:
    return [max(0.0, min(1000.0, float(y))), max(0.0, min(1000.0, float(x)))]


def sort_corners(points: list[list[float]]) -> list[list[float]]:
    by_y = sorted(points, key=lambda p: (p[0], p[1]))
    top = sorted(by_y[:2], key=lambda p: p[1])
    bottom = sorted(by_y[2:], key=lambda p: p[1])
    return [top[0], top[1], bottom[1], bottom[0]]


def force_parallelogram(points: list[list[float]]) -> list[list[float]]:
    if len(points) != 4:
        return points
    p0, p1, p2, _ = points
    y3 = p0[0] + p2[0] - p1[0]
    x3 = p0[1] + p2[1] - p1[1]
    return [p0, p1, p2, clamp_point(y3, x3)]


def parse_prediction(text: str, apply_geometry: bool) -> list[list[float]] | None:
    if not text or not text.strip():
        return None

    clean = text.strip()
    for marker in ("```json", "```"):
        if marker in clean:
            idx = clean.find(marker)
            rest = clean[idx + len(marker):].strip()
            end = rest.find("```")
            clean = rest[: end if end >= 0 else None].strip()

    match = re.search(r"\[[\s\S]*\]", clean)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    indexed: dict[int, list[float]] = {}
    unordered: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        point = item.get("point")
        if not isinstance(point, list) or len(point) < 2:
            continue
        parsed = clamp_point(point[0], point[1])
        label = str(item.get("label", "")).lower().strip()
        if label.startswith("corner_"):
            try:
                idx = int(label.split("_")[-1])
                indexed[idx] = parsed
                continue
            except ValueError:
                pass
        unordered.append(parsed)

    if len(indexed) == 4:
        points = [indexed[i] for i in range(4)]
    elif len(unordered) == 4:
        points = sort_corners(unordered)
    else:
        return None

    if apply_geometry:
        points = force_parallelogram(points)
    return points


def call_model(client: genai.Client, image_path: Path, prompt: str, model_name: str) -> tuple[list[list[float]] | None, str]:
    image_bytes = image_path.read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    raw_text = response.text or ""
    use_geometry = prompt == GEOMETRY_PROMPT
    return parse_prediction(raw_text, apply_geometry=use_geometry), raw_text


def point_error(pred: list[float], gt: list[float]) -> float:
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def mean_corner_error(prediction: list[list[float]], ground_truth: list[list[float]]) -> float:
    return sum(point_error(p, g) for p, g in zip(prediction, ground_truth)) / 4.0


def parallelogram_residual(points: list[list[float]]) -> float:
    left_y = points[0][0] + points[2][0]
    left_x = points[0][1] + points[2][1]
    right_y = points[1][0] + points[3][0]
    right_x = points[1][1] + points[3][1]
    return math.sqrt((left_y - right_y) ** 2 + (left_x - right_x) ** 2)


def draw_overlay(image_path: Path, gt_corners: list[list[float]], pred_corners: list[list[float]], save_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def to_px(point: list[float]) -> tuple[int, int]:
        return int(point[1] / 1000.0 * width), int(point[0] / 1000.0 * height)

    gt_px = [to_px(point) for point in gt_corners]
    pred_px = [to_px(point) for point in pred_corners]

    for idx in range(4):
        xg, yg = gt_px[idx]
        xp, yp = pred_px[idx]
        draw.ellipse((xg - 7, yg - 7, xg + 7, yg + 7), fill="lime")
        draw.ellipse((xp - 7, yp - 7, xp + 7, yp + 7), fill="red")
        draw.line((xg, yg, xp, yp), fill="yellow", width=2)
    image.save(save_path)


def save_results(rows: list[dict]) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"comparison_results_{timestamp}.json"
    csv_path = RESULTS_DIR / f"comparison_results_{timestamp}.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "mode",
                "mean_corner_error",
                "parallelogram_residual",
                "raw_text",
                "prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pointing comparison: baseline vs geometry-aware prompting.")
    parser.add_argument("--image", help="Run only one image name from corner_dataset.json", default=None)
    args = parser.parse_args()

    client = get_client()
    model_name = get_model_name()
    cases = load_cases()
    if args.image:
        cases = [case for case in cases if case.image_name == args.image]
        if not cases:
            raise ValueError(f"No case named {args.image} found in corner_dataset.json")

    rows: list[dict] = []
    for case in cases:
        image_path = DATASET_DIR / case.image_name
        for mode, prompt in (("baseline_visual_only", BASELINE_PROMPT), ("geometry_reasoning", GEOMETRY_PROMPT)):
            prediction, raw_text = call_model(client, image_path, prompt, model_name)
            if prediction is None:
                row = {
                    "image": case.image_name,
                    "mode": mode,
                    "mean_corner_error": None,
                    "parallelogram_residual": None,
                    "raw_text": raw_text,
                    "prediction": None,
                }
            else:
                overlay_path = RESULTS_DIR / f"{Path(case.image_name).stem}_{mode}_overlay.png"
                draw_overlay(image_path, case.gt_corners, prediction, overlay_path)
                row = {
                    "image": case.image_name,
                    "mode": mode,
                    "mean_corner_error": round(mean_corner_error(prediction, case.gt_corners), 4),
                    "parallelogram_residual": round(parallelogram_residual(prediction), 4),
                    "raw_text": raw_text,
                    "prediction": json.dumps(prediction, ensure_ascii=False),
                }
            rows.append(row)

    json_path, csv_path = save_results(rows)
    print(f"model={model_name}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    for row in rows:
        print(
            f"{row['image']} | {row['mode']} | "
            f"mean_corner_error={row['mean_corner_error']} | "
            f"parallelogram_residual={row['parallelogram_residual']}"
        )


if __name__ == "__main__":
    main()
