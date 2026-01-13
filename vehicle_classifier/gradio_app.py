import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import asyncio
import os

from config.config import Config
from llm.llm_analyzer import LLMAnalyzer


# ==============================
# 모델 & 설정 로드
# ==============================
config = Config()

MODEL_PATH = os.path.join(config.MODELS_DIR, "best_ResNet50.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

print(f"✓ 모델 로드 중: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["SUV", "버스", "세단", "승합", "이륜차", "트럭", "해치백", "화물"]

llm = LLMAnalyzer()


# ==============================
# 이미지 전처리
# ==============================
def preprocess_image(img: Image.Image):
    img = img.resize(config.IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ==============================
# Gradio에서 실행할 함수
# ==============================
def predict_and_explain(image):
    # CNN 예측
    img = preprocess_image(image)
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    vehicle_type = CLASS_NAMES[idx]
    confidence = float(preds[idx])

    # 모델별 점수 구조 (확장 대비)
    model_scores = {
        "ResNet50": float(confidence)
    }

    # LLM 분석 호출 (비동기 → 동기 실행)
    explanation = asyncio.run(
        llm.analyze_vehicle(
            predicted_class=vehicle_type,
            confidence=confidence,
            model_scores=model_scores
        )
    )

    summary = f"🚗 예측 차종: {vehicle_type} ({confidence*100:.2f}%)"

    return summary, explanation


# ==============================
# Gradio UI
# ==============================
interface = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="차량 이미지 업로드"),
    outputs=[
        gr.Textbox(label="CNN 예측 결과"),
        gr.Textbox(label="AI 분석 & 마케팅 리포트", lines=15)
    ],
    title="🚗 AI 차량 분석 & 마케팅 인텔리전스 시스템",
    description="""
이미지를 업로드하면 AI가 차종을 분석하고,
기술적 해석 + 고객 분석 + 마케팅 전략 + 광고 문구를 생성합니다.
"""
)

if __name__ == "__main__":
    interface.launch()
