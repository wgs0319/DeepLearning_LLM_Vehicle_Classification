"""
모델 학습 결과 시각화 유틸리티
- Loss / Accuracy 곡선
- 모델 성능 비교 그래프
"""

import os
import matplotlib.pyplot as plt
from config.config import Config


class Visualizer:
    def __init__(self):
        self.config = Config()
        self.save_dir = self.config.PLOTS_DIR

        # 저장 폴더 보장
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 한글 폰트 설정 (Windows 기준)
        plt.rcParams['font.family'] = self.config.FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

    # --------------------------------------------------
    # 1️⃣ 단일 모델 학습 곡선
    # --------------------------------------------------
    def plot_training_history(self, history, model_name):
        """
        loss / accuracy 학습 곡선 시각화
        """
        epochs = range(1, len(history.history['loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history['loss'], label='Train Loss')
        plt.plot(epochs, history.history['val_loss'], label='Val Loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f'{model_name}_training_curve.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ 학습 곡선 저장: {save_path}")

    # --------------------------------------------------
    # 2️⃣ 모든 모델 학습 곡선 생성
    # --------------------------------------------------
    def plot_all_histories(self, histories: dict):
        """
        여러 모델의 학습 곡선 생성
        """
        print("\n📈 학습 곡선 생성 중...")

        for model_name, history in histories.items():
            self.plot_training_history(history, model_name)

        print("✓ 모든 학습 곡선 생성 완료")

    # --------------------------------------------------
    # 3️⃣ 모델 성능 비교 (Validation Accuracy)
    # --------------------------------------------------
    def plot_model_comparison(self, results: dict):
        """
        모델 간 최고 Validation Accuracy 비교
        """
        model_names = list(results.keys())
        accuracies = [
            results[m]['best_val_accuracy'] for m in model_names
        ]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies)
        plt.title('모델별 최고 Validation Accuracy 비교')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        # 수치 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )

        plt.xticks(rotation=20)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'model_accuracy_comparison.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ 모델 성능 비교 그래프 저장: {save_path}")

    # --------------------------------------------------
    # 4️⃣ 전체 시각화 실행
    # --------------------------------------------------
    def generate_all_plots(self, histories: dict, results: dict):
        """
        모든 시각화 한 번에 생성
        """
        print("\n" + "=" * 60)
        print(" 📊 시각화 생성 시작")
        print("=" * 60)

        self.plot_all_histories(histories)
        self.plot_model_comparison(results)

        print("\n✓ 모든 시각화 완료")
