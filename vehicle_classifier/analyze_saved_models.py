"""
만들어진 모델에 대한 llm 분석기
"""

import os
import tensorflow as tf
from data.data_loader import DataLoader
from llm.llm_analyzer import LLMAnalyzer
import asyncio

MODEL_DIR = "dataset/models"

def load_models():
    models = {}

    for file in os.listdir(MODEL_DIR):
        if file.endswith(".keras"):
            name = file.replace(".keras", "")
            path = os.path.join(MODEL_DIR, file)
            print(f"📦 모델 로딩: {name}")
            models[name] = tf.keras.models.load_model(path)

    return models


def evaluate_models(models, val_data):
    results = {}

    for name, model in models.items():
        print(f"\n🔍 {name} 평가 중...")
        loss, acc = model.evaluate(val_data, verbose=0)

        results[name] = {
            "best_val_loss": float(loss),
            "best_val_accuracy": float(acc)
        }

        print(f"  Val Loss: {loss:.4f}")
        print(f"  Val Acc : {acc:.4f}")

    return results


async def run_llm_analysis(results):
    analyzer = LLMAnalyzer()

    print("\n🤖 Groq LLM 분석 중...")
    analysis = await analyzer.analyze_model_performance(results)

    os.makedirs("dataset/reports", exist_ok=True)
    report_path = "dataset/reports/saved_models_analysis.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(analysis)

    print(f"\n✅ LLM 분석 리포트 저장: {report_path}")
    print("\n" + analysis)


def main():
    print("="*60)
    print("📊 저장된 CNN 모델 분석 시작")
    print("="*60)

    # 1. 데이터 로드 (검증용)
    loader = DataLoader()
    _, val_data, class_names = loader.load_data()

    # 2. 모델 로드
    models = load_models()

    if not models:
        print("❌ dataset/models 폴더에 .keras 모델이 없습니다.")
        return

    # 3. 성능 평가
    results = evaluate_models(models, val_data)

    # 4. LLM 분석
    asyncio.run(run_llm_analysis(results))


if __name__ == "__main__":
    main()
