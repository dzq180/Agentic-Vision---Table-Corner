"""
基于 Gemini Robotics-ER 1.5，在 Blender 正交相机拍摄的图中精确定位桌面四个角点。
利用正交投影下桌面呈平行四边形的几何性质，结合 pointing 与推理（含遮挡角点的几何推算）。
对 table1 / table2 / table3 / table4 进行测试；table.png 为红点角点标记示例参考。
"""
import os
import re
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
# 测试用图：table1–4；table.png 为红点角点标记示例（若有则可选）
TABLE_IMAGES = ["table1.png", "table2.jpg", "table3.jpg", "table4.png", "table.png", "table5.jpg"]

# 正交相机 + 平行四边形 + 只考虑上桌面 + 三条线段相交定义角点 + 几何推算第四点 + 桌子特征 + 桌角强化
PROMPT = """This image was taken with an orthographic camera (e.g. from Blender). In orthographic projection the table surface appears as a parallelogram (no perspective distortion).

Your task: precisely locate the 4 corner points of the **tabletop only** (上桌面). Do not use points on the floor, legs, or objects on the table — only the corners of the top surface.

Table features that help locate the tabletop (use these as hints):
- **Flat horizontal surface**: The tabletop is one flat plane; its boundary in the image is a closed quadrilateral. The four corners are the only four points where two tabletop edges meet.
- **Sharp edge**: The tabletop edge is where the horizontal surface meets the vertical side or leg — look for this clear boundary line (color or brightness change, or a visible crease).
- **Legs**: Table legs usually meet the tabletop near the corners. The vertical line of a leg can help you find where the top surface ends; the corner lies where two tabletop edges meet, often near where a leg meets the top.
- **Objects on the table**: Cups, plates, etc. sit on top of the surface; the table extends under them. The corners are on the outer boundary of the table, not at the objects — ignore object outlines.
- **Uniform region**: The tabletop often has a roughly uniform color or texture inside; the boundary is where this surface ends (e.g. meeting the floor, wall, or leg).
- **Shadow/highlight**: A shadow or highlight along the table edge can make the boundary line easier to see.

How to identify a true table corner:
- A corner of the tabletop is where **three line segments meet**: the two edges of the tabletop that form that corner, and the vertical edge of the table at that corner (or the meeting of three visible edges at one point). Look for the intersection of these three segments — this uniquely defines a tabletop corner and avoids confusing other points.
- Only consider points that lie strictly on the **top surface** of the table (上桌面). This way you can reliably identify at least three visible corners.

**Corner order (tied to image coordinates — use this to avoid mixing corners):**
In the image, use the table outline’s position to assign corner_0, corner_1, corner_2, corner_3 unambiguously:
- **corner_0** = the vertex of the table outline that is **closest to the top-left** of the image (smallest y, then smallest x among the 4 corners). So in normalized coordinates, this corner has the smallest y; if two corners share the same y, pick the one with smaller x.
- **corner_1** = the vertex **closest to the top-right** (smallest y, largest x).
- **corner_2** = the vertex **closest to the bottom-right** (largest y, largest x).
- **corner_3** = the vertex **closest to the bottom-left** (largest y, smallest x).

Then the four corners are in order: corner_0 → corner_1 → corner_2 → corner_3, going **clockwise** around the table in the image (top-left, top-right, bottom-right, bottom-left). The "first" table edge is the segment from corner_0 to corner_1 (top edge), the "second" from corner_1 to corner_2 (right), the "third" from corner_2 to corner_3 (bottom), the "fourth" from corner_3 to corner_0 (left).

**Distinguishing the four corners (each corner has a different set of three meeting segments):**
At each corner, **exactly two** table edges meet plus the **vertical** edge of the table. The **direction** of these three segments is different at each corner — use this to confirm you have the right vertex:
- **corner_0** (top-left): where the **top** edge (corner_0–corner_1) meets the **left** edge (corner_3–corner_0), plus the vertical at that corner.
- **corner_1** (top-right): where the **top** edge meets the **right** edge (corner_1–corner_2), plus the vertical.
- **corner_2** (bottom-right): where the **right** edge meets the **bottom** edge (corner_2–corner_3), plus the vertical.
- **corner_3** (bottom-left): where the **bottom** edge meets the **left** edge, plus the vertical.

So corner_0 ≠ corner_1 ≠ corner_2 ≠ corner_3: each is the unique vertex with a different pair of table edges (and vertical) meeting. Output each point with the correct label so that corner_0 has smallest y (and smallest x if tie), corner_1 smallest y and largest x, etc.

**Stronger corner cues (use these to refine your choice):**
- **Edge endpoints**: Each corner is where **two tabletop edges meet**. Mentally follow each visible table edge until it ends — that endpoint is a corner. There are exactly 4 such endpoints forming the table outline.
- **Not object corners**: Ignore corners of objects ON the table (cup rim, book corner, plate edge). The table corner is on the **table's own** boundary, where the table surface meets the air or the leg.
- **Sharp turn on boundary**: At a true corner the table boundary makes a **sharp turn** (two edges meet). Any point in the middle of a long straight edge is NOT a corner.
- **Convex quadrilateral**: The 4 corners are the 4 vertices of the tabletop outline — a single convex quadrilateral with no crossing. They are the extremal points of the table surface in the image.
- **Do not use**: The center of the table, the center of any object on the table, or any point on the table leg below the table surface. Only the 4 vertices of the tabletop boundary.

Strategy:
1. Identify the 4 corners using the image-based order above (corner_0 = top-left, corner_1 = top-right, corner_2 = bottom-right, corner_3 = bottom-left). If one corner is occluded, compute it from the other three: corner_3 = corner_0 + (corner_2 - corner_1), i.e. (y3, x3) = (y0 + y2 - y1, x0 + x2 - x1).
2. Assign each point the correct label so that corner_0 has smallest y (then smallest x), corner_1 smallest y and largest x, corner_2 largest y and largest x, corner_3 largest y and smallest x.
3. Coordinates: normalized 0-1000, image top-left = (0,0), bottom-right = (1000,1000). Use [y, x] for each point.

Output: Return ONLY a JSON array of exactly 4 objects, no markdown, no explanation. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]

Critical: The four points MUST form a strict parallelogram (平行四边形). In other words, opposite sides must be parallel: corner_0, corner_1, corner_2, corner_3 in order must satisfy corner_0 + corner_2 = corner_1 + corner_3. If you output 4 points, ensure this relation holds; the standard way is to compute the 4th point from the first three as corner_3 = corner_0 + corner_2 - corner_1.
"""

# 使用参考图时的提示：第一张为带红点标记的示例，第二张为待检测图；角点顺序与主提示一致
PROMPT_WITH_REFERENCE = """You are given TWO images.

**First image**: A reference example. The red dots mark the 4 corner points of the tabletop (上桌面). Use this to see exactly what we mean by "table corner" — where two table edges meet on the top surface, plus the vertical at that corner. The order of the red dots in the reference matches the corner labels below.

**Second image**: The image in which you must find the 4 tabletop corners. Use the **same corner order** as in the main task (so your output is consistent and comparable):
- **corner_0** = vertex of the table outline **closest to the top-left** of the image (smallest y, then smallest x among the 4 corners).
- **corner_1** = **top-right** (smallest y, largest x).
- **corner_2** = **bottom-right** (largest y, largest x).
- **corner_3** = **bottom-left** (largest y, smallest x).

So the order is: corner_0 → corner_1 → corner_2 → corner_3, clockwise in the image (top-left, top-right, bottom-right, bottom-left). Each corner is where a **different pair** of table edges meet (plus the vertical); the three meeting segments have different directions at each corner — do not swap labels.

**Corner identification for the second image:**
- Each corner = **two tabletop edges meet** (follow each edge to its endpoint). Only the table's own boundary vertices; ignore corners of objects on the table (cup, book, plate).
- The 4 corners form one convex quadrilateral. At each corner the boundary makes a sharp turn. Do not use the center of the table or any point on the leg below the surface.
- If the 4th corner is occluded, compute it as corner_3 = corner_0 + corner_2 - corner_1. The four points MUST form a parallelogram.

Return the 4 corners for the **second image only**. Coordinates normalized 0-1000, [y, x] per point. Assign the correct label to each point so that corner_0 has smallest y (then smallest x), corner_1 smallest y and largest x, corner_2 largest y and largest x, corner_3 largest y and smallest x.

Output ONLY a JSON array, no markdown. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]
"""

REFERENCE_IMAGE_NAME = "table.png"


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 GEMINI_API_KEY。请在终端执行: $env:GEMINI_API_KEY=\"你的密钥\""
        )
    return genai.Client(api_key=api_key)


def _normalize_point(item: dict) -> tuple[float, float] | None:
    point = (
        item.get("point")
        or item.get("coordinates")
        or item.get("location")
    )
    if point is None and "x" in item and "y" in item:
        point = [item["y"], item["x"]]
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _parse_point_list_with_labels(data: list) -> list[tuple[tuple[float, float], int | None]]:
    """解析为 [(point, corner_index)]，corner_index 为 0–3 或 None。"""
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pt = _normalize_point(item)
        if pt is None:
            continue
        label = item.get("label") or item.get("name") or item.get("id")
        idx = None
        if label is not None:
            s = str(label).strip().lower()
            if s in ("corner_0", "corner0", "0"):
                idx = 0
            elif s in ("corner_1", "corner1", "1"):
                idx = 1
            elif s in ("corner_2", "corner2", "2"):
                idx = 2
            elif s in ("corner_3", "corner3", "3"):
                idx = 3
        out.append((pt, idx))
    return out


def parse_four_corners(text: str) -> list[tuple[float, float]] | None:
    """解析返回的 4 个角点 [y,x]，归一化 0-1000。若只有 3 个点则用平行四边形几何补全第 4 点。"""
    if not text or not text.strip():
        return None
    text = text.strip()

    for marker in ("```json", "```"):
        if marker in text:
            idx = text.find(marker)
            rest = text[idx + len(marker) :].strip()
            end = rest.find("```")
            text = rest[: end if end >= 0 else None].strip()

    # 找最外层 [...]
    bracket_start = text.find("[")
    if bracket_start < 0:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return _parse_corner_list(data)
            return None
        except json.JSONDecodeError:
            return None

    depth = 0
    for j, c in enumerate(text[bracket_start:], bracket_start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                json_str = text[bracket_start : j + 1]
                json_str = re.sub(r",\s*]", "]", json_str)
                json_str = re.sub(r",\s*}", "}", json_str)
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        return _parse_corner_list(data)
                except json.JSONDecodeError:
                    pass
                break
    return None


def _parse_corner_list(data: list) -> list[tuple[float, float]] | None:
    """解析角点列表：若存在 corner_0..3 的 label 则按 label 排序后再几何补全，否则按数组顺序。"""
    with_labels = _parse_point_list_with_labels(data)
    points_only = [p for p, _ in with_labels]
    if len(points_only) < 3:
        return None
    # 若 4 个点且 label 恰好为 0,1,2,3 各一，则按 label 排序
    indices = [i for _, i in with_labels if i is not None]
    if len(points_only) == 4 and set(indices) == {0, 1, 2, 3} and len(indices) == 4:
        by_idx = {i: p for p, i in with_labels if i is not None}
        points_only = [by_idx[k] for k in (0, 1, 2, 3)]
    return _ensure_four_corners(points_only)


def _parse_point_list(data: list) -> list[tuple[float, float]]:
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pt = _normalize_point(item)
        if pt is not None:
            out.append(pt)
    return out


def _ensure_four_corners(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """若恰好 4 个点则强制为平行四边形（用前三点推算第四点）；若 3 个点则用平行四边形几何补全第 4 点。"""
    if len(points) == 3:
        y0, x0 = points[0]
        y1, x1 = points[1]
        y2, x2 = points[2]
        y3 = y0 + y2 - y1
        x3 = x0 + x2 - x1
        return [points[0], points[1], points[2], (y3, x3)]
    if len(points) == 4:
        # 强制构成平行四边形：用 corner_0, corner_1, corner_2 推算 corner_3
        y0, x0 = points[0]
        y1, x1 = points[1]
        y2, x2 = points[2]
        y3 = y0 + y2 - y1
        x3 = x0 + x2 - x1
        return [points[0], points[1], points[2], (y3, x3)]
    return None


def draw_four_corners(image: Image.Image, corners: list[tuple[float, float]]) -> Image.Image:
    """在图像上绘制 4 个角点（红点）并连成红色四边形。corners 为 [y,x] 归一化 0-1000。"""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def to_px(y_norm: float, x_norm: float):
        return int(x_norm / 1000 * w), int(y_norm / 1000 * h)

    pts_px = [to_px(y, x) for y, x in corners]
    radius = max(6, min(w, h) // 100)
    width = max(2, min(w, h) // 300)

    for i, (x_px, y_px) in enumerate(pts_px):
        draw.ellipse(
            [x_px - radius, y_px - radius, x_px + radius, y_px + radius],
            outline="#FF0000",
            width=width,
            fill="#FF0000",
        )
    for i in range(4):
        x1, y1 = pts_px[i]
        x2, y2 = pts_px[(i + 1) % 4]
        draw.line([(x1, y1), (x2, y2)], fill="#FF0000", width=width)
    return img


def run_gemini_corner_detection(
    client,
    image_bytes: bytes,
    mime_type: str,
    reference_image_bytes: bytes | None = None,
    reference_image_mime: str | None = None,
) -> tuple[list[tuple[float, float]] | None, str]:
    """调用 Gemini 检测桌面四角点。若提供 reference_image，则先传参考图再传待检测图，用参考图辅助定位。"""
    if reference_image_bytes is not None and reference_image_mime:
        contents = [
            types.Part.from_bytes(data=reference_image_bytes, mime_type=reference_image_mime),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT_WITH_REFERENCE,
        ]
    else:
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ]

    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    raw_text = response.text or ""
    return parse_four_corners(raw_text), raw_text


def get_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def main():
    st.set_page_config(page_title="桌面四角点检测", layout="wide")
    st.title("正交相机桌面四角点定位（Gemini Robotics-ER 1.5）")

    available = [n for n in TABLE_IMAGES if (BASE_DIR / n).exists()]
    if not available:
        st.error(f"未找到桌面图像。请将 table1.png / table2.jpg 等放在: {BASE_DIR}")
        return

    selected = st.selectbox("选择图像", available, key="table_select")
    image_path = BASE_DIR / selected
    mime_type = get_mime(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(image_path).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(image)

    reference_available = (BASE_DIR / REFERENCE_IMAGE_NAME).exists() and selected != REFERENCE_IMAGE_NAME
    use_reference = st.checkbox(
        "使用参考图 table.png 辅助定位（将参考图与当前图一并发送，帮助模型理解“桌角点”含义）",
        value=True,
        disabled=not reference_available,
        key="use_ref",
    )
    if reference_available and not use_reference:
        st.caption("勾选后会在请求中附带 table.png，模型可参考红点位置理解桌角定义。")

    with col2:
        st.subheader("四角点检测（红点 + 红色四边形）")
        if st.button("运行 Gemini 四角点检测", type="primary"):
            with st.spinner("正在调用 Gemini（含几何推理）..."):
                try:
                    client = get_client()
                    ref_bytes = None
                    ref_mime = None
                    if use_reference and reference_available:
                        with open(BASE_DIR / REFERENCE_IMAGE_NAME, "rb") as f:
                            ref_bytes = f.read()
                        ref_mime = get_mime(BASE_DIR / REFERENCE_IMAGE_NAME)
                    corners, raw_text = run_gemini_corner_detection(
                        client, image_bytes, mime_type,
                        reference_image_bytes=ref_bytes,
                        reference_image_mime=ref_mime,
                    )
                    if corners and len(corners) == 4:
                        marked_image = draw_four_corners(image, corners)
                        st.image(marked_image)
                        st.json({
                            "corners": [
                                {"point": [round(c[0], 2), round(c[1], 2)], "label": f"corner_{i}"}
                                for i, c in enumerate(corners)
                            ],
                            "说明": "归一化坐标 [y, x]，0-1000",
                        })
                        stem = image_path.stem
                        output_path = BASE_DIR / f"{stem}_table_marked.png"
                        marked_image.save(output_path)
                        st.success(f"已保存到新文件: {output_path}（原图未修改）")
                    else:
                        st.warning("未解析到有效的 4 个角点")
                        with st.expander("查看 API 原始返回", expanded=True):
                            st.code(raw_text or "(空)", language="json")
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("点击上方按钮开始检测桌面四角点")

    st.caption(
        "正交相机下桌面为平行四边形；利用几何性质可推算被遮挡角点。"
        " table.png 为红点角点标记示例。"
    )


if __name__ == "__main__":
    main()
