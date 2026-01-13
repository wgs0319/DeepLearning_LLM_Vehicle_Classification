from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from config.config import Config


class CNNModels:
    def __init__(self):
        self.config = Config()
        self.INPUT_SHAPE = self.config.INPUT_SHAPE
        self.NUM_CLASSES = self.config.NUM_CLASSES
        self.LR = self.config.LEARNING_RATE

    # --------------------------------------------------
    # 1. CNN Basic
    # --------------------------------------------------
    def build_basic(self):
        model = models.Sequential([
            layers.Input(shape=self.INPUT_SHAPE),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # --------------------------------------------------
    # 2. CNN Deep
    # --------------------------------------------------
    def build_deep(self):
        model = models.Sequential([
            layers.Input(shape=self.INPUT_SHAPE),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # --------------------------------------------------
    # 3. CNN with Dropout (너의 베스트 모델)
    # --------------------------------------------------
    def build_dropout(self):
        model = models.Sequential([
            layers.Input(shape=self.INPUT_SHAPE),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # --------------------------------------------------
    # 4. main.py에서 호출하는 함수
    # --------------------------------------------------
    def get_all_models(self):
        return {
            "CNN_Basic": self.build_basic(),
            "CNN_Deep": self.build_deep(),
            "CNN_With_Dropout": self.build_dropout()
        }
