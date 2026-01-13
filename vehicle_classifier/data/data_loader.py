"""
데이터 로딩 및 전처리 모듈
- ImageDataGenerator를 사용한 실시간 데이터 증강
- 훈련/검증 데이터 자동 로드
- 클래스 정보 추출
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import Config
import os

class DataLoader:
    """
    차량 이미지 데이터 로더
    
    사용 방법:
        loader = DataLoader()
        train_data, val_data, class_names = loader.load_data()
    """
    
    def __init__(self):
        """초기화 및 설정 로드"""
        self.config = Config()
        self.train_generator = None
        self.val_generator = None
        self.train_data = None
        self.val_data = None
        self.class_names = None
        
        # 데이터 경로 확인
        self._validate_paths()
        
    def _validate_paths(self):
        """데이터 경로가 존재하는지 확인"""
        if not os.path.exists(self.config.BASE_DIR):
            raise FileNotFoundError(
                f"❌ 데이터 디렉토리를 찾을 수 없습니다: {self.config.BASE_DIR}\n"
                f"   다음 경로에 이미지 폴더가 있는지 확인하세요."
            )
        
        if not os.path.exists(self.config.TRAIN_DIR):
            raise FileNotFoundError(
                f"❌ 훈련 데이터 폴더를 찾을 수 없습니다: {self.config.TRAIN_DIR}"
            )
        
        if not os.path.exists(self.config.VAL_DIR):
            raise FileNotFoundError(
                f"❌ 검증 데이터 폴더를 찾을 수 없습니다: {self.config.VAL_DIR}"
            )
        
        print("✓ 데이터 경로 확인 완료")
        
    def create_generators(self):
        """
        ImageDataGenerator 생성
        
        훈련 데이터: 데이터 증강 적용
        검증 데이터: 정규화만 적용
        """
        print("\n데이터 증강 설정 중...")
        
        # 훈련 데이터 증강
        # - 회전, 이동, 확대/축소, 뒤집기로 데이터 다양성 증가
        # - 과적합 방지 효과
        self.train_generator = ImageDataGenerator(
            rescale=1./255,                              # [0, 255] → [0, 1] 정규화
            rotation_range=self.config.ROTATION_RANGE,   # ±15도 회전
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,   # 좌우 10% 이동
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE, # 상하 10% 이동
            zoom_range=self.config.ZOOM_RANGE,           # 10% 확대/축소
            horizontal_flip=self.config.HORIZONTAL_FLIP  # 좌우 반전
        )
        
        # 검증 데이터 (증강 없음, 정규화만)
        # - 실제 성능을 측정하기 위해 원본 그대로 사용
        self.val_generator = ImageDataGenerator(rescale=1./255)
        
        print("✓ 데이터 증강 설정 완료")
        print(f"  - 회전 범위: ±{self.config.ROTATION_RANGE}도")
        print(f"  - 이동 범위: {self.config.WIDTH_SHIFT_RANGE*100}%")
        print(f"  - 줌 범위: ±{self.config.ZOOM_RANGE*100}%")
        print(f"  - 좌우 반전: {'예' if self.config.HORIZONTAL_FLIP else '아니오'}")
        
    def load_data(self):
        """
        데이터 로딩
        
        폴더 구조 예시:
        images/vehicle_images/
        ├── Train/
        │   ├── SUV/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   ├── 버스/
        │   └── 세단/
        └── Validation/
            ├── SUV/
            ├── 버스/
            └── 세단/
        
        Returns:
            train_data: 훈련 데이터 제너레이터
            val_data: 검증 데이터 제너레이터
            class_names: 클래스 이름 리스트
        """
        # Generator가 없으면 생성
        if self.train_generator is None:
            self.create_generators()
        
        print(f"\n데이터 로딩 중...")
        print(f"  훈련 데이터 경로: {self.config.TRAIN_DIR}")
        print(f"  검증 데이터 경로: {self.config.VAL_DIR}")
        
        # 훈련 데이터 로드
        # flow_from_directory: 폴더 구조에서 자동으로 클래스 레이블 생성
        self.train_data = self.train_generator.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=self.config.IMG_SIZE,    # 이미지 크기 조정 (224x224)
            batch_size=self.config.BATCH_SIZE,   # 배치 크기 (32)
            class_mode='categorical',            # 다중 클래스 분류
            color_mode='rgb',                    # RGB 컬러
            shuffle=True,                        # 데이터 섞기 (훈련 시 중요!)
            seed=42                              # 재현성을 위한 시드
        )
        
        # 검증 데이터 로드
        self.val_data = self.val_generator.flow_from_directory(
            self.config.VAL_DIR,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False,                       # 검증 시에는 섞지 않음
            seed=42
        )
        
        # 클래스 이름 저장 (폴더 이름이 클래스명이 됨)
        self.class_names = list(self.train_data.class_indices.keys())
        
        # 로딩 결과 출력
        print(f"\n✓ 데이터 로딩 완료!")
        print(f"{'='*60}")
        print(f"  훈련 샘플 수: {self.train_data.samples:,}개")
        print(f"  검증 샘플 수: {self.val_data.samples:,}개")
        print(f"  총 샘플 수: {self.train_data.samples + self.val_data.samples:,}개")
        print(f"{'='*60}")
        print(f"  클래스 수: {len(self.class_names)}개")
        print(f"  클래스 목록: {', '.join(self.class_names)}")
        print(f"{'='*60}")
        print(f"  이미지 크기: {self.config.IMG_SIZE}")
        print(f"  배치 크기: {self.config.BATCH_SIZE}")
        print(f"  총 배치 수 (훈련): {len(self.train_data)}")
        print(f"  총 배치 수 (검증): {len(self.val_data)}")
        print(f"{'='*60}")
        
        return self.train_data, self.val_data, self.class_names
    
    def get_data(self):
        """
        이미 로드된 데이터 반환 (없으면 자동 로드)
        
        Returns:
            train_data, val_data, class_names
        """
        if self.train_data is None:
            self.load_data()
        return self.train_data, self.val_data, self.class_names
    
    def get_class_distribution(self):
        """
        클래스별 샘플 수 확인
        
        Returns:
            dict: {클래스명: 샘플수}
        """
        if self.train_data is None:
            self.load_data()
        
        # 훈련 데이터의 클래스별 샘플 수
        train_distribution = {}
        for class_name, class_idx in self.train_data.class_indices.items():
            count = sum(self.train_data.classes == class_idx)
            train_distribution[class_name] = count
        
        print(f"\n📊 훈련 데이터 클래스 분포:")
        print(f"{'='*60}")
        for class_name, count in sorted(train_distribution.items()):
            percentage = (count / self.train_data.samples) * 100
            print(f"  {class_name:<15}: {count:>5}개 ({percentage:>5.1f}%)")
        print(f"{'='*60}")
        
        return train_distribution
    
    def preview_batch(self, num_images=5):
        """
        배치 샘플 미리보기
        
        Args:
            num_images: 표시할 이미지 수
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if self.train_data is None:
            self.load_data()
        
        # 한 배치 가져오기
        images, labels = next(self.train_data)
        
        # 시각화
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        
        for i in range(min(num_images, len(images))):
            axes[i].imshow(images[i])
            
            # 레이블 찾기
            label_idx = np.argmax(labels[i])
            label_name = self.class_names[label_idx]
            
            axes[i].set_title(f'{label_name}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('./model/data_preview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 샘플 이미지 저장: ./model/data_preview.png")


# 사용 예시
if __name__ == "__main__":
    """
    데이터 로더 테스트 및 사용 예시
    """
    print("="*60)
    print(" 데이터 로더 테스트")
    print("="*60)
    
    try:
        # 1. 데이터 로더 초기화
        loader = DataLoader()
        
        # 2. 데이터 로드
        train_data, val_data, class_names = loader.load_data()
        
        # 3. 클래스 분포 확인
        loader.get_class_distribution()
        
        # 4. 샘플 미리보기
        print("\n샘플 이미지 생성 중...")
        loader.preview_batch(num_images=5)
        
        # 5. 데이터 정보 출력
        print(f"\n✓ 테스트 완료!")
        print(f"\n사용 가능한 데이터:")
        print(f"  - train_data: {type(train_data)}")
        print(f"  - val_data: {type(val_data)}")
        print(f"  - class_names: {class_names}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("\n다음을 확인하세요:")
        print("  1. 데이터 폴더가 올바른 경로에 있는지")
        print("  2. 폴더 구조가 다음과 같은지:")
        print("     images/vehicle_images/")
        print("     ├── Train/")
        print("     │   ├── SUV/")
        print("     │   ├── 버스/")
        print("     │   └── ...")
        print("     └── Validation/")
        print("         ├── SUV/")
        print("         ├── 버스/")
        print("         └── ...")