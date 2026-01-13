"""
모델 훈련 및 비교 로직
- 단일 / 다중 모델 학습
- EarlyStopping, ModelCheckpoint 적용
- 모델 성능 비교 결과 정리
"""

import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config.config import Config


class ModelTrainer:
    """
    모델 훈련을 담당하는 클래스
    """

    def __init__(self):
        self.config = Config()
        self.histories = {}   # 모델별 history 저장
        self.results = {}     # 모델별 성능 요약 저장

        # dataset/models 폴더 보장
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

    # --------------------------------------------------
    # Callback 생성
    # --------------------------------------------------
    def _get_callbacks(self, model_name: str):
        """
        콜백 생성 (EarlyStopping + ModelCheckpoint)
        """
        checkpoint_path = os.path.join(
            self.config.MODELS_DIR,
            f"{model_name}.keras"   # 항상 최신 모델로 덮어쓰기
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        return [early_stopping, model_checkpoint]

    # --------------------------------------------------
    # 단일 모델 학습
    # --------------------------------------------------
    def train_model(self, model, model_name, train_data, val_data):
        """
        단일 모델 훈련
        """
        print("\n" + "=" * 60)
        print(f" 🚀 모델 학습 시작: {model_name}")
        print("=" * 60)

        callbacks = self._get_callbacks(model_name)

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        self.histories[model_name] = history

        print(f"\n✓ {model_name} 학습 완료")
        print(f"✓ 모델 저장 위치: {os.path.join(self.config.MODELS_DIR, model_name + '.keras')}")

        return history

    # --------------------------------------------------
    # 여러 모델 학습
    # --------------------------------------------------
    def train_multiple_models(self, models_dict, train_data, val_data):
        """
        여러 모델을 순차적으로 학습
        """
        print("\n" + "=" * 60)
        print(" 📌 다중 모델 학습 시작")
        print("=" * 60)

        for model_name, model in models_dict.items():
            self.train_model(
                model=model,
                model_name=model_name,
                train_data=train_data,
                val_data=val_data
            )

        print("\n✓ 모든 모델 학습 완료")
        return self.histories

    # --------------------------------------------------
    # 모델 성능 비교
    # --------------------------------------------------
    def compare_models(self):
        """
        학습된 모델들의 성능 비교 요약
        """
        print("\n" + "=" * 60)
        print(" 📊 모델 성능 비교")
        print("=" * 60)

        if not self.histories:
            raise ValueError("학습된 모델이 없습니다. train_multiple_models()를 먼저 실행하세요.")

        for model_name, history in self.histories.items():
            val_acc = history.history.get("val_accuracy", [])
            val_loss = history.history.get("val_loss", [])

            self.results[model_name] = {
                "best_val_accuracy": float(max(val_acc)) if val_acc else None,
                "final_val_accuracy": float(val_acc[-1]) if val_acc else None,
                "best_val_loss": float(min(val_loss)) if val_loss else None,
                "final_val_loss": float(val_loss[-1]) if val_loss else None,
                "trained_epochs": len(val_loss)
            }

            print(f"\n🔹 {model_name}")
            print(f"  - 최고 Val Accuracy : {self.results[model_name]['best_val_accuracy']:.4f}")
            print(f"  - 최종 Val Accuracy : {self.results[model_name]['final_val_accuracy']:.4f}")
            print(f"  - 최소 Val Loss     : {self.results[model_name]['best_val_loss']:.4f}")
            print(f"  - 학습 Epoch 수     : {self.results[model_name]['trained_epochs']}")

        print("\n✓ 모델 비교 완료")
        return self.results

    # --------------------------------------------------
    # Getter
    # --------------------------------------------------
    def get_histories(self):
        return self.histories

    def get_results(self):
        return self.results
