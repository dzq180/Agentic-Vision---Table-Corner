import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated_3d_scenes"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_OPTIONS = [
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
]

STYLE_PRESETS = {
    "通用三维渲染": "high-quality 3D render, spatially coherent scene, realistic depth, detailed lighting, clean composition",
    "等距视角": "isometric 3D scene, 45-degree top-down view, miniature spatial layout, clean geometry, clear depth separation",
    "室内设计效果图": "architectural interior render, photorealistic materials, soft indirect lighting, balanced composition, realistic shadows",
    "科幻空间": "futuristic 3D environment, cinematic volumetric lighting, layered depth, polished materials, immersive atmosphere",
    "低多边形": "low-poly 3D scene, clean shapes, stylized geometry, pleasant colors, readable object silhouettes",
}

LIGHTING_PRESETS = {
    "自然白天": "natural daylight, soft global illumination",
    "黄昏电影感": "golden hour lighting, cinematic contrast, warm highlights",
    "夜景霓虹": "night scene with neon accents, moody lighting, reflective surfaces",
    "工作室打光": "studio lighting, controlled highlights, crisp shadows",
}


def _relay_base_url() -> str | None:
    for key in ("GOOGLE_GEMINI_BASE_URL", "GEMINI_BASE_URL", "GOOGLE_GENAI_BASE_URL"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip().rstrip("/")
    return None


def _default_model() -> str:
    preferred = os.environ.get("GEMINI_IMAGE_MODEL") or os.environ.get("GEMINI_MODEL")
    if preferred in MODEL_OPTIONS:
        return preferred
    return MODEL_OPTIONS[0]


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY. Set it in PowerShell first, for example:\n"
            '$env:GEMINI_API_KEY="your-key"'
        )

    base_url = _relay_base_url()
    if base_url:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(base_url=base_url),
        )
    return genai.Client(api_key=api_key)


def build_generation_prompt(
    user_prompt: str,
    style_preset: str,
    lighting_preset: str,
    extra_requirements: str,
) -> str:
    preset_text = STYLE_PRESETS[style_preset]
    lighting_text = LIGHTING_PRESETS[lighting_preset]
    prompt = (
        "请生成一张具有清晰三维空间感的场景图片。"
        "画面必须明确体现景深、物体之间的位置关系、遮挡关系和整体空间结构。"
        f"风格要求：{preset_text}。"
        f"光照要求：{lighting_text}。"
        "整体画面需要视觉统一、结构合理、空间关系可信，并具有较强的三维表现力。"
        "最终效果应接近高质量三维渲染图或空间场景概念图。"
        f"场景描述：{user_prompt.strip()}。"
    )
    if extra_requirements.strip():
        prompt += f"补充要求：{extra_requirements.strip()}。"
    return prompt


def _image_from_inline_data(inline_data) -> Image.Image:
    data = getattr(inline_data, "data", None)
    if data is None:
        raise ValueError("The model returned an image part without binary image data.")
    return Image.open(BytesIO(data)).convert("RGB")


def _extract_response_parts(response) -> list:
    if getattr(response, "parts", None):
        return list(response.parts)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        if content and getattr(content, "parts", None):
            return list(content.parts)
    return []


def _pil_image_to_part(image: Image.Image) -> types.Part:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def generate_3d_scene(
    client: genai.Client,
    model_name: str,
    prompt: str,
    reference_image: Image.Image | None = None,
) -> tuple[Image.Image, str]:
    contents: list = []
    if reference_image is not None:
        contents.append(_pil_image_to_part(reference_image))
    contents.append(prompt)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    text_fragments: list[str] = []
    generated_image: Image.Image | None = None

    for part in _extract_response_parts(response):
        if getattr(part, "text", None):
            text_fragments.append(part.text)
        elif getattr(part, "inline_data", None) is not None and generated_image is None:
            generated_image = _image_from_inline_data(part.inline_data)

    if generated_image is None:
        raise ValueError("The model did not return an image. Try a different prompt or switch models.")

    return generated_image, "\n".join(text_fragments).strip()


def save_generated_image(image: Image.Image) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"generated_3d_scene_{timestamp}.png"
    image.save(output_path)
    return output_path


def main() -> None:
    st.set_page_config(page_title="三维空间图片生成器", layout="wide")
    st.title("提示词生成三维空间图片")
    st.caption(
        "这是一个位于数据集文件夹中的本地 Streamlit 页面，可根据提示词生成三维空间场景图片，也支持上传参考图继续生成。"
    )

    relay_url = _relay_base_url()
    if relay_url:
        st.caption(f"当前 Gemini 接口地址：`{relay_url}`")

    with st.sidebar:
        st.subheader("生成设置")
        model_name = st.selectbox(
            "图像模型",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(_default_model()),
        )
        style_preset = st.selectbox("风格预设", list(STYLE_PRESETS.keys()), index=0)
        lighting_preset = st.selectbox("光照预设", list(LIGHTING_PRESETS.keys()), index=0)
        st.markdown("环境变量")
        st.code(
            '$env:GEMINI_API_KEY="your-key"\n'
            '$env:GEMINI_IMAGE_MODEL="gemini-2.5-flash-image"',
            language="powershell",
        )

    default_prompt = (
        "一个未来感室内展厅，中央有悬浮平台和机器人机械臂，"
        "空间层次分明，材质精致，整体具有明显的三维纵深感。"
    )
    user_prompt = st.text_area("输入提示词", value=default_prompt, height=140)
    extra_requirements = st.text_area(
        "补充要求",
        value="保持结构清晰、主体明确，并让透视关系和遮挡关系足够明显。",
        height=90,
    )
    uploaded = st.file_uploader(
        "参考图（可选）",
        type=["png", "jpg", "jpeg", "webp"],
    )

    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.subheader("最终生成提示词")
        final_prompt = build_generation_prompt(
            user_prompt=user_prompt,
            style_preset=style_preset,
            lighting_preset=lighting_preset,
            extra_requirements=extra_requirements,
        )
        st.code(final_prompt, language="text")

    reference_image = None
    with preview_cols[1]:
        st.subheader("参考图预览")
        if uploaded is not None:
            reference_image = Image.open(uploaded).convert("RGB")
            st.image(reference_image, use_container_width=True)
        else:
            st.info("如果不上传参考图，将仅根据文本提示词生成图片。")

    if "generated_image_bytes" not in st.session_state:
        st.session_state.generated_image_bytes = None
    if "generated_image_name" not in st.session_state:
        st.session_state.generated_image_name = None
    if "generated_model_text" not in st.session_state:
        st.session_state.generated_model_text = ""

    if st.button("生成三维空间图片", type="primary"):
        try:
            with st.spinner("正在调用 Gemini 生成图片..."):
                client = get_client()
                generated_image, model_text = generate_3d_scene(
                    client=client,
                    model_name=model_name,
                    prompt=final_prompt,
                    reference_image=reference_image,
                )
                buffer = BytesIO()
                generated_image.save(buffer, format="PNG")
                st.session_state.generated_image_bytes = buffer.getvalue()
                st.session_state.generated_model_text = model_text
                st.session_state.generated_image_name = (
                    f"generated_3d_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )

            result_cols = st.columns(2)
            with result_cols[0]:
                st.subheader("生成结果")
                st.image(generated_image, use_container_width=True)
            with result_cols[1]:
                st.subheader("结果信息")
                if model_text:
                    st.write("模型文本输出")
                    st.code(model_text, language="text")
                st.info("当前结果仅预览，点击下方“保存当前图片”后才会写入 generated_3d_scenes 文件夹。")
                st.download_button(
                    "下载 PNG",
                    data=st.session_state.generated_image_bytes,
                    file_name=st.session_state.generated_image_name,
                    mime="image/png",
                )
            st.success("图片已生成，当前仅用于预览，尚未保存到本地文件夹。")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.generated_image_bytes is not None:
        st.markdown("已生成图片")
        preview_image = Image.open(BytesIO(st.session_state.generated_image_bytes)).convert("RGB")
        st.image(preview_image, use_container_width=True)
        if st.button("保存当前图片"):
            saved_image = Image.open(BytesIO(st.session_state.generated_image_bytes)).convert("RGB")
            output_path = save_generated_image(saved_image)
            st.success(f"图片已保存到：`{output_path}`")
        if st.session_state.generated_model_text:
            with st.expander("查看本次模型文本输出"):
                st.code(st.session_state.generated_model_text, language="text")

    st.markdown("运行命令")
    st.code(
        "streamlit run 数据集/prompt_to_3d_scene_app.py",
        language="powershell",
    )


if __name__ == "__main__":
    main()
