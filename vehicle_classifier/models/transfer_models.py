"""
Transfer Learning 모델 정의
- ResNet50
- MobileNetV2
"""

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from config.config import Config


class TransferModels:
    """
    Transfer Learning 모델 관리 클래스
    """

    def __init__(self):
        self.learning_rate = Config.LEARNING_RATE

    # ==============================
    # 01. ResNet50
    # ==============================
    def build_resnet50(self):
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=Config.INPUT_SHAPE
        )

        # 1단계: 사전학습 가중치 동결
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(Config.NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ==============================
    # 02. MobileNetV2
    # ==============================
    def build_mobilenet(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=Config.INPUT_SHAPE
        )

        # 1단계: 사전학습 가중치 동결
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(Config.NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ==============================
    # 모든 Transfer 모델 반환
    # ==============================
    def get_all_models(self):
        """
        Transfer Learning 모델들을 dict 형태로 반환
        → ModelTrainer.train_multiple_models 에서 사용
        """
        return {
            "ResNet50": self.build_resnet50(),
            "MobileNetV2": self.build_mobilenet()
        }

    # ==============================
    # preprocess 함수 제공 (중요)
    # ==============================
    def get_preprocess_functions(self):
        """
        모델별 preprocess_input 반환
        → DataLoader / Trainer에서 사용
        """
        return {
            "ResNet50": resnet_preprocess,
            "MobileNetV2": mobilenet_preprocess
        }
