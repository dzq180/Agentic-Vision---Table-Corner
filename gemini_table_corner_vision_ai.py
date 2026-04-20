"""
在 Blender 正交相机拍摄的图中精确定位桌子的四个角点。
目标不变（识别桌子四角），方法上：(1) 增强定位已有像素：坐标裁剪、按位置排序、提示词要求小数精确定位，可选 OpenCV 边缘细化；
(2) 已有信息+几何常识+数学推理：桌面为平面→正交下呈平行四边形，对角线交于中心，三点推第四点，代码强制平行四边形（可与「仅大模型」管线对比）；
(3) 分割模型技术：可选「分割+Gemini」——用 OpenCV 轮廓检测分割桌面区域，凸包取四极值点后平行四边形拟合得到四角点（仅在与「含数学几何推理」同选时启用）。
对 table1 / table2 / table3 / table4 等测试；table.png 为红点角点标记示例。
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
# 测试图集中放在「数据集」子目录；若无该目录则仍从脚本同目录读取（兼容旧布局）
DATASET_DIR = BASE_DIR / "数据集"
IMAGE_ROOT = DATASET_DIR if DATASET_DIR.is_dir() else BASE_DIR
# 测试用图：table1–4；table.png 为红点角点标记示例（若有则可选）
TABLE_IMAGES = ["table1.png", "table2.jpg", "table3.jpg", "table4.jpg", "table.png", "table5.jpg"]

# 目标：桌子四角点；方法：强调桌子具有的几何性质，用几何与数学推理定位
PROMPT = """This image was taken with an orthographic camera (e.g. from Blender). Your task: **locate the 4 corner points of the table** (the tabletop only). Use the **geometric properties of the table** to find them.

**Geometric properties of the table (use these to reason):**
- The **tabletop is a flat horizontal plane**. In orthographic projection there is no perspective distortion: a flat rectangle in 3D projects to a **parallelogram** in the image (possibly a rectangle). So the table's 4 corners are exactly the **4 vertices of that parallelogram**.
- A parallelogram has 4 vertices; each vertex is where **two edges of the tabletop boundary meet**. Find the four edges that form the tabletop outline; their endpoints are the table corners.
- **Parallelogram condition** (in image coordinates): if we label the 4 corners in order corner_0, corner_1, corner_2, corner_3, then opposite sides are parallel iff corner_0 + corner_2 = corner_1 + corner_3. If **one corner is occluded**, do NOT guess its position from object edges — instead, use the other three corners and this equation to COMPUTE the missing corner exactly:
  - missing corner_0: corner_0 = corner_1 + corner_3 − corner_2
  - missing corner_1: corner_1 = corner_0 + corner_2 − corner_3
  - missing corner_2: corner_2 = corner_1 + corner_3 − corner_0
  - missing corner_3: corner_3 = corner_0 + corner_2 − corner_1
  For example, if only corner_0, corner_1, corner_2 are clearly visible and corner_3 is fully occluded, you MUST output corner_3 = corner_0 + corner_2 − corner_1, i.e. (y3, x3) = (y0 + y2 − y1, x0 + x2 − x1).

Use these geometric properties to identify the table's 4 corners: find the parallelogram that is the tabletop, then output its 4 vertices. Ignore corners of objects on the table; only the table's own boundary.

**Further geometry (for reasoning or verification):**
- The **diagonals** of the table (corner_0–corner_2 and corner_1–corner_3) **intersect at the center** of the parallelogram. Use this to check or infer positions.
- If **only three corners** are clearly visible, output those three with correct labels; the fourth is uniquely determined by corner_3 = corner_0 + corner_2 - corner_1.

**Order by position in the image:**
- corner_0 = vertex **closest to top-left** (smallest y; if tie, smallest x).
- corner_1 = **top-right** (smallest y, largest x).
- corner_2 = **bottom-right** (largest y, largest x).
- corner_3 = **bottom-left** (largest y, smallest x).

**Precise localization:** Each corner is the **intersection of two straight table edges**. Report coordinates as decimals in 0–1000 (e.g. one decimal place) for precise pixel-level localization. Image top-left = (0,0), bottom-right = (1000,1000). Use [y, x] for each point.

Output: Return ONLY a JSON array of exactly 4 objects, no markdown, no explanation. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]

The four points MUST form a strict parallelogram (the tabletop in this view). Prefer computing the 4th point from the first three with corner_3 = corner_0 + corner_2 - corner_1 so the condition holds exactly.
"""

# 对比实验：仅依赖视觉与模型输出，不在提示词中要求平行四边形/向量公式推理
PROMPT_LLM_ONLY = """This image was taken with an orthographic camera (e.g. from Blender). Your task: **locate the 4 corner points of the table** (the tabletop only — vertices where the top surface outline meets; ignore corners of objects on the table).

Use **pure visual judgment**: estimate each tabletop corner in the image. Do **not** use or mention parallelogram equations, vector sums, or computing a missing corner from three points; output only what you infer from pixels.

**Order by position in the image:**
- corner_0 = vertex closest to top-left (smallest y; if tie, smallest x).
- corner_1 = top-right (smallest y, largest x among the upper corners).
- corner_2 = bottom-right (largest y, largest x).
- corner_3 = bottom-left (largest y, smallest x).

**Coordinates:** Decimals in 0–1000 for sub-pixel feel. Image top-left = (0,0), bottom-right = (1000,1000). Use [y, x] for each point.

Output: Return ONLY a JSON array of exactly 4 objects, no markdown, no explanation. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]
"""

# 参考图模式：第一张为桌角红点示例，第二张用桌子的几何性质（平行四边形）推理出四角
PROMPT_WITH_REFERENCE = """You are given TWO images.

**First image**: A reference. The red dots mark the **4 corner points of the table** (tabletop). The table has the geometric property that its top surface is a flat plane, so in orthographic view it appears as a parallelogram; the red dots are the 4 vertices of that parallelogram. The order matches: corner_0 = top-left, corner_1 = top-right, corner_2 = bottom-right, corner_3 = bottom-left.

**Second image**: Locate the **4 corner points of the table** in this image. Use the **same geometric properties**: the tabletop is a flat plane → in this view it is a parallelogram; the 4 table corners are the 4 vertices; diagonals intersect at center; if only 3 corners visible, the 4th = corner_0 + corner_2 - corner_1. Report coordinates as decimals (0–1000) for precise localization. Order by position: corner_0 = top-left, corner_1 = top-right, corner_2 = bottom-right, corner_3 = bottom-left.

Return the 4 table corners for the **second image only**. Coordinates normalized 0-1000, [y, x] per point. Output ONLY a JSON array, no markdown. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]
"""

PROMPT_WITH_REFERENCE_LLM_ONLY = """You are given TWO images.

**First image**: A reference. The red dots mark the **4 corner points of the table** (tabletop). Order: corner_0 = top-left, corner_1 = top-right, corner_2 = bottom-right, corner_3 = bottom-left.

**Second image**: Locate the **4 corner points of the table** in this image using **visual judgment only**, analogous to the first image. Do **not** use parallelogram equations or compute hidden corners from formulas; output coordinates you infer from the second image pixels.

Coordinates: decimals 0–1000, [y, x] per point, same ordering as above. Return the 4 corners for the **second image only**. Output ONLY a JSON array, no markdown. Format:
[{"point": [y, x], "label": "corner_0"}, {"point": [y, x], "label": "corner_1"}, {"point": [y, x], "label": "corner_2"}, {"point": [y, x], "label": "corner_3"}]
"""

REFERENCE_IMAGE_NAME = "table.png"

# 中转站配置：从环境变量读取，未设置则用默认
def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


def _relay_base_url() -> str | None:
    """中转 Gemini 兼容接口的根地址（不要末尾 /）。支持多个常见环境变量名。"""
    for key in (
        "GOOGLE_GEMINI_BASE_URL",
        "GEMINI_BASE_URL",
        "GOOGLE_GENAI_BASE_URL",
    ):
        url = os.environ.get(key)
        if url and str(url).strip():
            return str(url).strip().rstrip("/")
    return None


def get_client():
    """使用 GEMINI_API_KEY；若设置中转 base URL（_relay_base_url）则请求发往中转站。

    中转站密钥多为 sk- 开头，与官方 Google AI 密钥格式不同，可直接作为 GEMINI_API_KEY。
    base URL 须与服务商文档一致（须兼容 google-genai 调用的 Gemini API 路径，不是 OpenAI 的 /v1/chat）。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 GEMINI_API_KEY。\n"
            "PowerShell 示例（中转站）：\n"
            '  $env:GEMINI_API_KEY="sk-你的令牌"\n'
            '  $env:GOOGLE_GEMINI_BASE_URL="https://控制台域名"   # 按服务商「Gemini / google-genai」文档填写，无尾斜杠\n'
            "也可用 GEMINI_BASE_URL，与 GOOGLE_GEMINI_BASE_URL 等价。\n"
            "模型名需与中转可用列表一致，例如: $env:GEMINI_MODEL=\"gemini-2.0-flash\""
        )
    base_url = _relay_base_url()
    if base_url:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(base_url=base_url),
        )
    return genai.Client(api_key=api_key)


def _clamp_normalized(y: float, x: float) -> tuple[float, float]:
    """将归一化坐标限制在 [0, 1000]，避免越界与绘图异常。"""
    return (
        max(0.0, min(1000.0, float(y))),
        max(0.0, min(1000.0, float(x))),
    )


def _normalize_point(item: dict) -> tuple[float, float] | None:
    point = (
        item.get("point")
        or item.get("coordinates")
        or item.get("location")
    )
    if point is None and "x" in item and "y" in item:
        point = [item["y"], item["x"]]
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        y, x = float(point[0]), float(point[1])
        return _clamp_normalized(y, x)
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


def parse_four_corners(text: str, *, apply_geometry: bool = True) -> list[tuple[float, float]] | None:
    """解析返回的 4 个角点 [y,x]，归一化 0-1000。

    apply_geometry=True：解析返回的 4 个角点 [y,x]，归一化 0-1000。若只有 3 个点则用平行四边形几何补全第 4 点。
    apply_geometry=False：仅大模型对比模式，不补点、不强制平行四边形，须恰好解析到 4 个点。
    """
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
                return _parse_corner_list(data, apply_geometry=apply_geometry)
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
                        return _parse_corner_list(data, apply_geometry=apply_geometry)
                except json.JSONDecodeError:
                    pass
                break
    return None


def _parse_corner_list(
    data: list, *, apply_geometry: bool = True
) -> list[tuple[float, float]] | None:
    """解析角点列表：若有 corner_0..3 的 label 则按 label排序；否则 4 点时按图像位置排序，再几何补全。。

    apply_geometry 为 True 时再做 _ensure_four_corners（含 3→4 补点）；为 False 时仅接受模型给出的 4 点，不强制平行四边形。
    """
    with_labels = _parse_point_list_with_labels(data)
    points_only = [p for p, _ in with_labels]
    min_len = 3 if apply_geometry else 4
    if len(points_only) < min_len:
        return None

    indices = [i for _, i in with_labels if i is not None]
    by_idx = {i: p for p, i in with_labels if i is not None}

    if len(points_only) == 4 and set(indices) == {0, 1, 2, 3} and len(indices) == 4:
        points_only = [by_idx[k] for k in (0, 1, 2, 3)]
    elif len(points_only) == 4:
        points_only = _sort_four_points_by_position(points_only)
    elif not apply_geometry and len(points_only) > 4:
        if set(by_idx.keys()) == {0, 1, 2, 3}:
            points_only = [by_idx[k] for k in (0, 1, 2, 3)]
        else:
            return None

    if apply_geometry:
        return _ensure_four_corners(points_only)
    if len(points_only) != 4:
        return None
    return points_only


def _parse_point_list(data: list) -> list[tuple[float, float]]:
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pt = _normalize_point(item)
        if pt is not None:
            out.append(pt)
    return out


def _sort_four_points_by_position(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """将 4 个点按图像位置排序为：左上、右上、右下、左下（corner_0..3），便于几何补全时顺序一致。"""
    if len(points) != 4:
        return points
    by_y = sorted(points, key=lambda p: (p[0], p[1]))
    top_two = sorted(by_y[:2], key=lambda p: p[1])
    bottom_two = sorted(by_y[2:], key=lambda p: p[1])
    top_left, top_right = top_two[0], top_two[1]
    bottom_left, bottom_right = bottom_two[0], bottom_two[1]
    return [top_left, top_right, bottom_right, bottom_left]


def _ensure_four_corners(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """若恰好 4 个点则强制为平行四边形（用前三点推算第四点）；若 3 个点则用平行四边形几何补全第 4 点。"""
    if len(points) == 3:
        y0, x0 = points[0]
        y1, x1 = points[1]
        y2, x2 = points[2]
        y3 = y0 + y2 - y1
        x3 = x0 + x2 - x1
        return [points[0], points[1], points[2], _clamp_normalized(y3, x3)]
    if len(points) == 4:
        # 强制构成平行四边形：用 corner_0, corner_1, corner_2 推算 corner_3
        y0, x0 = points[0]
        y1, x1 = points[1]
        y2, x2 = points[2]
        y3 = y0 + y2 - y1
        x3 = x0 + x2 - x1
        return [points[0], points[1], points[2], _clamp_normalized(y3, x3)]
    return None


def _opencv_available() -> bool:
    try:
        import cv2
        import numpy as np
        return True
    except ImportError:
        return False


def _four_corners_from_hull_stable(pts, h: int, w: int) -> list[tuple[float, float]] | None:
    """从凸包点集用稳定方式取四角：argmin/argmax(y±x)，避免排序歧义。pts 为 Nx2 (x,y)。"""
    import numpy as np
    if len(pts) < 4:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    y_vals = pts[:, 1]
    x_vals = pts[:, 0]
    y_plus_x = y_vals + x_vals
    y_minus_x = y_vals - x_vals
    idx_tl = int(np.argmin(y_plus_x))   # 左上: y+x 最小
    idx_tr = int(np.argmin(y_minus_x))  # 右上: y-x 最小
    idx_br = int(np.argmax(y_plus_x))   # 右下: y+x 最大
    idx_bl = int(np.argmax(y_minus_x))   # 左下: y-x 最大
    if len({idx_tl, idx_tr, idx_br, idx_bl}) < 4:
        return None
    top_left = pts[idx_tl]
    top_right = pts[idx_tr]
    bottom_right = pts[idx_br]
    bottom_left = pts[idx_bl]
    four = [top_left, top_right, bottom_right, bottom_left]
    norm = []
    for px, py in four:
        ny = float(py) / h * 1000.0
        nx = float(px) / w * 1000.0
        norm.append(_clamp_normalized(ny, nx))
    return _ensure_four_corners(norm)


def segment_table_and_get_corners(image: Image.Image) -> list[tuple[float, float]] | None:
    """用分割（轮廓检测）+ 平行四边形拟合得到桌面四角点。需 OpenCV。失败返回 None。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    w, h = image.size
    if w < 10 or h < 10:
        return None
    gray = np.array(image.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    combined = cv2.bitwise_or(edges, thresh)
    # 形态学闭运算：弥合细小缝隙，使同一桌面得到稳定单一轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    min_area = (w * h) * 0.02
    # 稳定排序：面积降序，面积相同时按外接矩形 (y,x) 升序，保证同图多次结果一致
    def contour_sort_key(c):
        area = cv2.contourArea(c)
        x_rect, y_rect, _, _ = cv2.boundingRect(c)
        return (-area, y_rect, x_rect)
    contours = sorted(contours, key=contour_sort_key)
    best_result = None
    best_score = -1.0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        hull = cv2.convexHull(cnt)
        if len(hull) < 4:
            continue
        out = _four_corners_from_hull_stable(hull, h, w)
        if not out:
            continue
        # 选“最像桌面”的轮廓：面积大且凸包紧凑（面积/凸包面积比高）优先
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        if hull_area <= 0:
            continue
        ratio = cnt_area / hull_area
        score = cnt_area * (0.5 + 0.5 * ratio)
        if score > best_score:
            best_score = score
            best_result = out
    return best_result


def _refine_corners_with_edges(
    image: Image.Image, corners: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """用边缘+亚像素角点细化已知角点位置（需 OpenCV）。

    步骤：
    1. 在角点附近用 Canny 找最近的边缘点（粗调到真实边界上）。
    2. 以该点为初值调用 cv2.cornerSubPix 做亚像素角点 refine。
    最终输出仍做归一化裁剪，避免越界。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return corners
    if len(corners) != 4:
        return corners
    w, h = image.size
    gray = np.array(image.convert("L"))
    gray_f = gray.astype("float32")
    edges = cv2.Canny(gray, 50, 150)
    out = []
    # 在角点附近小窗口内找边缘点，然后用 cornerSubPix 做亚像素 refine
    radius = max(5, min(w, h) // 60)
    for (y_norm, x_norm) in corners:
        x_px = int(x_norm / 1000 * w)
        y_px = int(y_norm / 1000 * h)
        x_px = max(radius, min(w - 1 - radius, x_px))
        y_px = max(radius, min(h - 1 - radius, y_px))
        roi = edges[
            y_px - radius : y_px + radius + 1,
            x_px - radius : x_px + radius + 1,
        ]
        ys, xs = np.where(roi > 0)
        if len(ys) == 0:
            # 没有明显边缘，保持原位置
            out.append((y_norm, x_norm))
            continue
        # 以原角点为原点，选最近的边缘点（在 roi 内坐标需加偏移）
        ys_abs = ys + (y_px - radius)
        xs_abs = xs + (x_px - radius)
        dists = (xs_abs - x_px) ** 2 + (ys_abs - y_px) ** 2
        i = int(np.argmin(dists))
        y_edge = int(ys_abs[i])
        x_edge = int(xs_abs[i])
        # cornerSubPix 需要 float32 角点输入，形状 (N,1,2)
        import cv2  # type: ignore[redefined-builtin]

        corners_init = np.array([[[x_edge, y_edge]]], dtype="float32")
        win_size = (radius, radius)
        zero_zone = (-1, -1)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        refined = cv2.cornerSubPix(gray_f, corners_init, win_size, zero_zone, crit)
        x_ref, y_ref = refined[0, 0]
        y_new = y_ref / h * 1000.0
        x_new = x_ref / w * 1000.0
        out.append(_clamp_normalized(y_new, x_new))
    return out


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
    *,
    use_geometry_pipeline: bool = True,
) -> tuple[list[tuple[float, float]] | None, str]:
    """调用 Gemini 检测桌面四角点。若提供 reference_image，则先传参考图再传待检测图，用参考图辅助定位。

    use_geometry_pipeline=False 时使用 PROMPT_LLM_ONLY / 参考图纯视觉版，解析不做平行四边形代码约束。
    """
    if reference_image_bytes is not None and reference_image_mime:
        prompt = (
            PROMPT_WITH_REFERENCE
            if use_geometry_pipeline
            else PROMPT_WITH_REFERENCE_LLM_ONLY
        )
        contents = [
            types.Part.from_bytes(data=reference_image_bytes, mime_type=reference_image_mime),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]
    else:
        prompt = PROMPT if use_geometry_pipeline else PROMPT_LLM_ONLY
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]

    response = client.models.generate_content(
        model=_model_name(),
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    raw_text = response.text or ""
    return parse_four_corners(raw_text, apply_geometry=use_geometry_pipeline), raw_text


def get_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def main():
    st.set_page_config(page_title="桌面四角点检测", layout="wide")
    st.title("正交相机桌面四角点定位（Gemini Robotics-ER 1.5）")
    relay_url = _relay_base_url()
    if relay_url:
        st.caption(
            "API：已启用中转（GOOGLE_GEMINI_BASE_URL / GEMINI_BASE_URL）。"
            f"模型：`{_model_name()}`。请在同一终端启动 streamlit 前设置环境变量。"
        )
    else:
        st.caption(f"API：直连 Google（未设置中转 BASE_URL）。模型：`{_model_name()}`。")

    available = [n for n in TABLE_IMAGES if (IMAGE_ROOT / n).exists()]
    if not available:
        st.error(
            f"未找到桌面图像。请将 table1.png / table2.jpg 等放在: {DATASET_DIR} "
            f"（或脚本同目录: {BASE_DIR}）"
        )
        return

    selected = st.selectbox("选择图像", available, key="table_select")
    image_path = IMAGE_ROOT / selected
    mime_type = get_mime(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(image_path).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(image)

    reference_available = (IMAGE_ROOT / REFERENCE_IMAGE_NAME).exists() and selected != REFERENCE_IMAGE_NAME
    use_reference = st.checkbox(
        "使用参考图 table.png 辅助定位（将参考图与当前图一并发送，帮助模型理解“桌角点”含义）",
        value=True,
        disabled=not reference_available,
        key="use_ref",
    )
    if reference_available and not use_reference:
        st.caption("勾选后会在请求中附带 table.png，模型可参考红点理解「桌子四角点」的输出格式与顺序。")

    detection_method = st.radio(
        "检测方式",
        options=[
            "仅 Gemini",
            "分割初定位 + Gemini 精修（推荐）",
        ],
        index=0,
        key="detection_method",
        horizontal=True,
    )
    use_segment_then_gemini = detection_method.startswith("分割初定位")
    if use_segment_then_gemini and not _opencv_available():
        st.warning("当前未安装 OpenCV（opencv-python、numpy），分割初定位功能不可用。请安装后重试：pip install opencv-python numpy")

    reasoning_mode = st.radio(
        "角点推理策略（对比实验）",
        options=[
            "含数学几何推理（提示词 + 代码平行四边形）",
            "仅大模型（无几何公式提示，无代码平行四边形 / 补点 / 边缘细化）",
        ],
        index=0,
        key="reasoning_mode",
        horizontal=True,
    )
    use_geometry_pipeline = reasoning_mode.startswith("含")
    if use_segment_then_gemini and not use_geometry_pipeline:
        st.info(
            "「仅大模型」模式下不使用分割初定位（分割含凸包等几何先验）。本次将自动改为仅调用 Gemini。"
        )
    effective_segment = use_segment_then_gemini and use_geometry_pipeline and _opencv_available()

    with col2:
        st.subheader("四角点检测（红点 + 红色四边形）")
        if st.button("运行检测", type="primary"):
            corners = None
            raw_text = ""
            try:

                def _post_process_gemini_corners(
                    pts: list[tuple[float, float]] | None,
                ) -> list[tuple[float, float]] | None:
                    if not pts or len(pts) != 4:
                        return pts
                    if not use_geometry_pipeline:
                        return pts
                    pts = _refine_corners_with_edges(image, pts)
                    return _ensure_four_corners(pts) or pts

                if effective_segment:
                    with st.spinner("先分割初定位，再调用 Gemini 精修..."):
                        seg_corners = segment_table_and_get_corners(image)
                        if seg_corners and len(seg_corners) == 4:
                            client = get_client()
                            ref_bytes = ref_mime = None
                            if use_reference and reference_available:
                                with open(IMAGE_ROOT / REFERENCE_IMAGE_NAME, "rb") as f:
                                    ref_bytes = f.read()
                                ref_mime = get_mime(IMAGE_ROOT / REFERENCE_IMAGE_NAME)
                            corners, raw_text = run_gemini_corner_detection(
                                client,
                                image_bytes,
                                mime_type,
                                reference_image_bytes=ref_bytes,
                                reference_image_mime=ref_mime,
                                use_geometry_pipeline=use_geometry_pipeline,
                            )
                            if corners and len(corners) == 4:
                                corners = _post_process_gemini_corners(corners)
                            else:
                                corners = seg_corners
                                raw_text = "（分割成功，Gemini 未返回有效角点，使用分割结果）"
                        else:
                            client = get_client()
                            ref_bytes = ref_mime = None
                            if use_reference and reference_available:
                                with open(IMAGE_ROOT / REFERENCE_IMAGE_NAME, "rb") as f:
                                    ref_bytes = f.read()
                                ref_mime = get_mime(IMAGE_ROOT / REFERENCE_IMAGE_NAME)
                            corners, raw_text = run_gemini_corner_detection(
                                client,
                                image_bytes,
                                mime_type,
                                reference_image_bytes=ref_bytes,
                                reference_image_mime=ref_mime,
                                use_geometry_pipeline=use_geometry_pipeline,
                            )
                            if corners and len(corners) == 4:
                                corners = _post_process_gemini_corners(corners)
                else:
                    spin = (
                        "正在调用 Gemini（含数学几何提示与后处理）..."
                        if use_geometry_pipeline
                        else "正在调用 Gemini（仅大模型：无几何公式提示与代码约束）..."
                    )
                    with st.spinner(spin):
                        client = get_client()
                        ref_bytes = ref_mime = None
                        if use_reference and reference_available:
                            with open(IMAGE_ROOT / REFERENCE_IMAGE_NAME, "rb") as f:
                                ref_bytes = f.read()
                            ref_mime = get_mime(IMAGE_ROOT / REFERENCE_IMAGE_NAME)
                        corners, raw_text = run_gemini_corner_detection(
                            client,
                            image_bytes,
                            mime_type,
                            reference_image_bytes=ref_bytes,
                            reference_image_mime=ref_mime,
                            use_geometry_pipeline=use_geometry_pipeline,
                        )
                        if corners and len(corners) == 4:
                            corners = _post_process_gemini_corners(corners)
                if corners and len(corners) == 4:
                    marked_image = draw_four_corners(image, corners)
                    st.image(marked_image)
                    out_suffix = "_table_marked_geom.png" if use_geometry_pipeline else "_table_marked_llm.png"
                    st.json({
                        "角点推理策略": reasoning_mode,
                        "corners": [
                            {"point": [round(c[0], 2), round(c[1], 2)], "label": f"corner_{i}"}
                            for i, c in enumerate(corners)
                        ],
                        "说明": "归一化坐标 [y, x]，0-1000",
                    })
                    stem = image_path.stem
                    output_path = image_path.parent / f"{stem}{out_suffix}"
                    marked_image.save(output_path)
                    st.success(f"已保存到新文件: {output_path}（原图未修改）")
                else:
                    st.warning("未解析到有效的 4 个角点")
                    if raw_text:
                        with st.expander("查看 API 原始返回", expanded=True):
                            st.code(raw_text or "(空)", language="json")
            except Exception as e:
                st.error(str(e))
        else:
            st.info("选择检测方式后点击「运行检测」开始")

    st.caption(
        "目标：识别桌子四角点。「含数学几何推理」在提示词与解析中利用平行四边形性质并可代码强制四角共面；"
        "「仅大模型」不做公式提示、不补点、不强制平行四边形、不做 OpenCV 边缘细化，便于对比。"
        "table.png 为红点角点参考示例。"
    )


if __name__ == "__main__":
    main()
