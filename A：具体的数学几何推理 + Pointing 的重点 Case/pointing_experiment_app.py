import json
import math
import os
import re
import ast
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from google import genai
from google.genai import types


WORK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORK_DIR.parent
DATASET_DIR = PROJECT_ROOT / "数据集"
GROUND_TRUTH_PATH = PROJECT_ROOT / "corner_dataset.json"

BASELINE_PROMPT = """You are given one image captured by an orthographic camera.
Task: locate the 4 tabletop corner points of the table.
Use visual judgment only. Do not use geometric equations, vector sums, parallelogram constraints, or derive a hidden corner from the other three corners.
Order: corner_0 top-left, corner_1 top-right, corner_2 bottom-right, corner_3 bottom-left.
Coordinates are normalized to 0-1000 and each point uses [y, x].
Return only a JSON array.
"""

GEOMETRY_PROMPT = """You are given one image captured by an orthographic camera.
Task: locate the 4 tabletop corner points of the table.
Use explicit geometric reasoning:
- the tabletop is a flat plane
- under orthographic projection it forms a parallelogram
- opposite sides are parallel
- corner_0 + corner_2 = corner_1 + corner_3
- if one corner is unclear, compute it from the other three instead of guessing
- diagonals intersect at the center
Order: corner_0 top-left, corner_1 top-right, corner_2 bottom-right, corner_3 bottom-left.
Coordinates are normalized to 0-1000 and each point uses [y, x].
Return only a JSON array.
"""


def _relay_base_url() -> str | None:
    for key in ("GOOGLE_GEMINI_BASE_URL", "GEMINI_BASE_URL", "GOOGLE_GENAI_BASE_URL"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip().rstrip("/")
    return None


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError('未设置 GEMINI_API_KEY，例如：$env:GEMINI_API_KEY="your-key"')
    base_url = _relay_base_url()
    if base_url:
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(base_url=base_url))
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def load_ground_truth() -> dict[str, list[list[float]]]:
    items = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return {item["image"]: item["corners"] for item in items}


def clamp_point(y: float, x: float) -> list[float]:
    return [max(0.0, min(1000.0, float(y))), max(0.0, min(1000.0, float(x)))]


def sort_corners(points: list[list[float]]) -> list[list[float]]:
    by_y = sorted(points, key=lambda p: (p[0], p[1]))
    top = sorted(by_y[:2], key=lambda p: p[1])
    bottom = sorted(by_y[2:], key=lambda p: p[1])
    return [top[0], top[1], bottom[1], bottom[0]]


def force_parallelogram(points: list[list[float]]) -> list[list[float]]:
    p0, p1, p2, _ = points
    y3 = p0[0] + p2[0] - p1[0]
    x3 = p0[1] + p2[1] - p1[1]
    return [p0, p1, p2, clamp_point(y3, x3)]


def _extract_candidate_arrays(text: str) -> list[str]:
    """提取文本中所有可能的 JSON 数组片段（按括号配对）。"""
    candidates: list[str] = []
    starts = [i for i, ch in enumerate(text) if ch == "["]
    for start in starts:
        depth = 0
        for idx, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _load_json_like_array(candidate: str):
    """优先按 JSON 解析，失败时用 literal_eval 容错（单引号等）。"""
    cleaned = re.sub(r",\s*]", "]", candidate)
    cleaned = re.sub(r",\s*}", "}", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return None


def parse_prediction(text: str, apply_geometry: bool) -> tuple[list[list[float]] | None, str]:
    if not text or not text.strip():
        return None, "模型返回为空"

    clean = text.strip()

    # 兼容 ```json ... ``` / ``` ... ``` 包裹
    lines = clean.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            clean = "\n".join(lines[i + 1 :]).split("```")[0].strip()
            break
    if "```" in clean and "[" not in clean[: clean.find("```")]:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if code_block:
            clean = code_block.group(1).strip()

    # 优先提取最外层 JSON 数组，避免正则贪婪导致解析失败
    bracket_start = clean.find("[")
    if bracket_start < 0:
        # 兼容 {"points":[...]} / {"items":[...]} 结构
        try:
            obj = json.loads(clean)
        except json.JSONDecodeError:
            return None, "模型返回既不是数组，也不含 points/items 字段"
        if isinstance(obj, dict):
            if isinstance(obj.get("points"), list):
                data = obj["points"]
            elif isinstance(obj.get("items"), list):
                data = obj["items"]
            else:
                return None, "对象结构中未找到 points/items 数组"
        elif isinstance(obj, list):
            data = obj
        else:
            return None, "模型返回顶层结构不是数组"
    else:
        data = None
        for candidate in _extract_candidate_arrays(clean):
            parsed = _load_json_like_array(candidate)
            if isinstance(parsed, list):
                # 优先选择“看起来像角点数组”的候选：元素为 dict 且含 point/x/y
                looks_like_points = any(
                    isinstance(it, dict)
                    and ("point" in it or ("x" in it and "y" in it))
                    for it in parsed
                )
                if looks_like_points:
                    data = parsed
                    break
                if data is None:
                    data = parsed
        if data is None:
            return None, "未找到可解析的数组片段"

    try:
        if not isinstance(data, list):
            return None, "解析后顶层不是数组"

        indexed: dict[int, list[float]] = {}
        unordered: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            point = item.get("point")
            if point is None and "x" in item and "y" in item:
                point = [item["y"], item["x"]]
            if not isinstance(point, list) or len(point) < 2:
                continue
            parsed = clamp_point(point[0], point[1])
            label = str(item.get("label", "")).lower().strip()
            if label.startswith("corner_"):
                try:
                    indexed[int(label.split("_")[-1])] = parsed
                    continue
                except ValueError:
                    pass
            unordered.append(parsed)

        if len(indexed) == 4:
            points = [indexed[i] for i in range(4)]
        elif len(unordered) == 4:
            points = sort_corners(unordered)
        else:
            return None, "数组中未解析到 4 个有效角点"

        return (force_parallelogram(points) if apply_geometry else points), "ok"
    except Exception as exc:
        return None, f"解析异常: {exc}"


def call_model(
    client: genai.Client, image_path: Path, prompt: str, model_name: str
) -> tuple[list[list[float]] | None, str, str]:
    image_bytes = image_path.read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    raw_text = response.text or ""
    prediction, reason = parse_prediction(raw_text, apply_geometry=(prompt == GEOMETRY_PROMPT))
    return prediction, raw_text, reason


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


def render_overlay(image_path: Path, gt_corners: list[list[float]] | None, pred_corners: list[list[float]] | None) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def to_px(point: list[float]) -> tuple[int, int]:
        return int(point[1] / 1000.0 * width), int(point[0] / 1000.0 * height)

    if gt_corners:
        for point in gt_corners:
            x, y = to_px(point)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="lime")
    if pred_corners:
        for point in pred_corners:
            x, y = to_px(point)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="red")
    return image


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="几何推理 Pointing 对比实验", layout="wide")
    st.title("数学几何推理对视觉定位的作用对比实验")
    st.caption("红点表示模型预测，绿点表示标注真值。目标是比较纯视觉提示与几何推理提示在桌角定位任务上的差异。")

    gt_map = load_ground_truth()
    image_names = [name for name in gt_map.keys() if (DATASET_DIR / name).exists()]
    selected = st.selectbox("选择实验图片", image_names)
    image_path = DATASET_DIR / selected
    gt_corners = gt_map[selected]

    st.sidebar.subheader("实验设置")
    st.sidebar.write(f"当前模型：`{get_model_name()}`")
    relay_url = _relay_base_url()
    if relay_url:
        st.sidebar.write(f"接口地址：`{relay_url}`")
    st.sidebar.info("建议使用同一个模型，分别运行 baseline 和 geometry，便于公平对比。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("原图")
        st.image(Image.open(image_path), use_container_width=True)

    if st.button("运行对比实验", type="primary"):
        try:
            client = get_client()
            baseline_pred, baseline_raw, baseline_reason = call_model(
                client, image_path, BASELINE_PROMPT, get_model_name()
            )
            geometry_pred, geometry_raw, geometry_reason = call_model(
                client, image_path, GEOMETRY_PROMPT, get_model_name()
            )

            with col2:
                st.subheader("Baseline：纯视觉判断")
                st.image(render_overlay(image_path, gt_corners, baseline_pred), use_container_width=True)
                if baseline_pred:
                    st.metric("平均角点误差", f"{mean_corner_error(baseline_pred, gt_corners):.2f}")
                    st.metric("平行四边形残差", f"{parallelogram_residual(baseline_pred):.2f}")
                    st.json(baseline_pred)
                else:
                    st.error("Baseline 未解析出有效角点。")
                    st.caption(f"解析原因：{baseline_reason}")
                with st.expander("查看 Baseline 原始输出"):
                    st.code(baseline_raw or "(empty)", language="text")

            with col3:
                st.subheader("Geometry：数学几何推理")
                st.image(render_overlay(image_path, gt_corners, geometry_pred), use_container_width=True)
                if geometry_pred:
                    st.metric("平均角点误差", f"{mean_corner_error(geometry_pred, gt_corners):.2f}")
                    st.metric("平行四边形残差", f"{parallelogram_residual(geometry_pred):.2f}")
                    st.json(geometry_pred)
                else:
                    st.error("Geometry 未解析出有效角点。")
                    st.caption(f"解析原因：{geometry_reason}")
                with st.expander("查看 Geometry 原始输出"):
                    st.code(geometry_raw or "(empty)", language="text")

            if baseline_pred and geometry_pred:
                delta = mean_corner_error(baseline_pred, gt_corners) - mean_corner_error(geometry_pred, gt_corners)
                if delta > 0:
                    st.success(f"几何推理版本的平均角点误差更低，提升了 {delta:.2f}。")
                elif delta < 0:
                    st.warning(f"当前图片上 baseline 更低，领先 {abs(delta):.2f}。建议继续扩展 case。")
                else:
                    st.info("当前图片上两者平均角点误差相同。")
        except Exception as exc:
            st.error(str(exc))

    st.markdown("运行方式")
    st.code("streamlit run .\\A：具体的数学几何推理 + Pointing 的重点 Case\\pointing_experiment_app.py", language="powershell")


if __name__ == "__main__":
    main()
