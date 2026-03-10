"""
使用 Gemini Robotics-ER 1.5 对 10_items.png 进行物体检测与标记，
并通过 Streamlit 在本地 URL 展示标记后的图像。
"""
import os
import re
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "10_items.png"

PROMPT = """
Point to no more than 10 items in the image. The label returned
should be an identifying name for the object detected.
The answer should follow the json format: [{"point": [y, x], "label": "<label>"}, ...].
The points are in [y, x] format normalized to 0-1000.
Return ONLY valid JSON, no other text.
"""


def get_client():
    """获取配置了 API Key 的 Gemini 客户端"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 GEMINI_API_KEY。请在终端执行: $env:GEMINI_API_KEY=\"你的密钥\""
        )
    return genai.Client(api_key=api_key)


def parse_gemini_response(text: str) -> list[dict]:
    """从 Gemini 返回的文本中解析 JSON 格式的 point/label 列表（兼容官方 cookbook 格式）"""
    if not text or not text.strip():
        return []

    text = text.strip()

    # 1. 按官方 cookbook 方式去除 markdown 代码块 ```json ... ```
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            text = "\n".join(lines[i + 1 :])
            text = text.split("```")[0].strip()
            break

    # 2. 若无 ```json，尝试通用代码块
    if "```" in text and "[" not in text[: text.find("```")]:
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block:
            text = code_block.group(1).strip()

    # 3. 查找最外层的 JSON 数组 [...]
    bracket_start = text.find("[")
    if bracket_start >= 0:
        depth = 0
        for j, c in enumerate(text[bracket_start:], bracket_start):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    json_str = text[bracket_start : j + 1]
                    # 去除尾随逗号等常见 LLM 输出问题
                    json_str = re.sub(r",\s*]", "]", json_str)
                    json_str = re.sub(r",\s*}", "}", json_str)
                    try:
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            return _normalize_points(data)
                    except json.JSONDecodeError:
                        pass
                    break

    # 4. 尝试解析整个文本
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _normalize_points(data)
        if isinstance(data, dict) and "points" in data:
            return _normalize_points(data["points"])
        if isinstance(data, dict) and "items" in data:
            return _normalize_points(data["items"])
    except json.JSONDecodeError:
        pass

    return []


def _normalize_points(data: list) -> list[dict]:
    """统一不同 API 返回格式为 {point: [y,x], label: str}"""
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # 支持 point / coordinates / location 等键名
        point = (
            item.get("point")
            or item.get("coordinates")
            or item.get("location")
            or item.get("pos")
        )
        label = (
            item.get("label")
            or item.get("name")
            or item.get("object")
            or item.get("description")
            or "item"
        )
        if point is None and "x" in item and "y" in item:
            point = [item["y"], item["x"]]
        if point is None and "y" in item and "x" in item:
            point = [item["y"], item["x"]]
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            result.append({"point": [float(point[0]), float(point[1])], "label": str(label)})
    return result


def draw_markers_on_image(image: Image.Image, points: list[dict]) -> Image.Image:
    """在图像上绘制检测到的点与标签"""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 尝试加载字体，失败则用默认
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
        except OSError:
            font = ImageFont.load_default()

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FF8800", "#8800FF", "#0088FF", "#88FF00",
    ]

    for i, item in enumerate(points):
        point = item.get("point")
        label = item.get("label", f"item_{i}")
        if not point or not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        y_norm, x_norm = float(point[0]), float(point[1])
        # 坐标归一化到 0-1000，转换为像素
        x_px = int(x_norm / 1000 * w)
        y_px = int(y_norm / 1000 * h)

        color = colors[i % len(colors)]
        radius = max(8, min(w, h) // 80)

        # 绘制圆点
        draw.ellipse(
            [x_px - radius, y_px - radius, x_px + radius, y_px + radius],
            outline=color,
            width=3,
            fill=color,
        )
        # 绘制标签
        draw.text((x_px + radius + 4, y_px - 8), str(label), fill=color, font=font)

    return img


def run_gemini_detection(client, image_bytes: bytes) -> tuple[list[dict], str]:
    """调用 Gemini Robotics-ER 进行检测，返回 (解析结果, 原始响应文本)"""
    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw_text = response.text or ""
    return parse_gemini_response(raw_text), raw_text


def main():
    st.set_page_config(page_title="Gemini Vision 标记", layout="wide")
    st.title("Gemini Robotics-ER 1.5 图像标记")

    if not IMAGE_PATH.exists():
        st.error(f"未找到图像文件: {IMAGE_PATH}")
        return

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图像")
        image = Image.open(IMAGE_PATH).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("标记后图像")
        if st.button("运行 Gemini 检测并标记", type="primary"):
            with st.spinner("正在调用 Gemini Robotics-ER 1.5..."):
                try:
                    client = get_client()
                    points, raw_text = run_gemini_detection(client, image_bytes)
                    if points:
                        marked_image = draw_markers_on_image(image, points)
                        st.image(marked_image, use_container_width=True)
                        st.json(points)
                        # 保存标记后的图像
                        output_path = BASE_DIR / "10_items_marked.png"
                        marked_image.save(output_path)
                        st.success(f"已保存到: {output_path}")
                    else:
                        st.warning("未解析到有效检测结果")
                        with st.expander("查看 API 原始返回（便于排查）", expanded=True):
                            st.code(raw_text or "(空)", language="json")
                            st.caption("若返回格式与预期不符，可将上述内容反馈以便改进解析逻辑")
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("点击上方按钮开始检测")

    st.caption("使用 gemini-robotics-er-1.5-preview 模型")


if __name__ == "__main__":
    main()
