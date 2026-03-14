import os

from google import genai
from google.genai import types

PROMPT = """
          Point to no more than 10 items in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """


def get_client() -> genai.Client:
    """使用环境变量；若设置 GOOGLE_GEMINI_BASE_URL 则走中转站。"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("未设置 GEMINI_API_KEY 环境变量")
    base_url = os.environ.get("GOOGLE_GEMINI_BASE_URL")
    if base_url:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(base_url=base_url.rstrip("/")),
        )
    return genai.Client(api_key=api_key)


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")


client = get_client()

# Load your image
with open("10_items.png", "rb") as f:
    image_bytes = f.read()

image_response = client.models.generate_content(
    model=_model_name(),
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        ),
        PROMPT,
    ],
    config=types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

print(image_response.text)
