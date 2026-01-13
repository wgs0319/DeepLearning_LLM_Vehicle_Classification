"""
프로젝트 전체 설정 및 하이퍼파라미터 관리
"""
import os

class Config:
    # 데이터 경로
    BASE_DIR = "images"
    TRAIN_DIR = os.path.join(BASE_DIR, "Train")
    VAL_DIR = os.path.join(BASE_DIR, "Validation")

    # 결과 자료 저장 경로
    DATASET_DIR = "dataset"

    # 모델 저장 경로
    MODELS_DIR = os.path.join(DATASET_DIR, "models")
    # 시각화 자료 저장 경로
    PLOTS_DIR = os.path.join(DATASET_DIR, "plots")
    # llm리포트 저장 경로
    REPORTS_DIR = os.path.join(DATASET_DIR, "reports")

    # 이미지 설정
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    
    # 훈련 하이퍼파라미터
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # 데이터 증강 파라미터
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    
    # EarlyStopping 설정
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.0
    
    # Matplotlib 한글 설정
    FONT_FAMILY = 'Malgun Gothic'
    
    @classmethod
    def create_model_dir(cls):
        """모델 저장 디렉토리 생성"""
        if not os.path.exists(cls.MODEL_DIR):
            os.makedirs(cls.MODEL_DIR)
            print(f"✓ 모델 저장 폴더 생성: {cls.MODEL_DIR}")