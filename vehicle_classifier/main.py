"""
차량 이미지 분류 프로젝트 메인 실행 파일
CNN → Transfer Learning → LLM 분석 + 결과 자동 저장
"""
import argparse
import os
from datetime import datetime

from config.config import Config
from data.data_loader import DataLoader
from models.cnn_models import CNNModels
from models.transfer_models import TransferModels
from training.trainer import ModelTrainer
from utils.visualization import Visualizer

# LLM 선택적 임포트
try:
    from llm.llm_analyzer_groq import GroqLLMAnalyzer
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq 패키지가 설치되지 않았습니다. LLM 기능을 사용하려면 'pip install groq'를 실행하세요.")


# ==============================
# 결과 폴더 생성 (고정 경로)
# ==============================
def create_experiment_dir():
    """
    dataset 폴더에 고정된 구조 생성 (덮어쓰기 방식)
    
    dataset/
    ├── models/
    ├── plots/
    └── reports/
    """
    exp_dir = Config.DATASET_DIR
    
    # dataset 폴더가 없으면 생성
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"✓ dataset 폴더 생성: {exp_dir}")
    
    # 하위 폴더 생성 (이미 있어도 OK)
    subdirs = ['models', 'plots', 'reports']
    for subdir in subdirs:
        subdir_path = os.path.join(exp_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
    
    print(f"\n📁 결과 저장 경로: {exp_dir}")
    print(f"   - 모델 저장: {exp_dir}/models/ (덮어쓰기)")
    print(f"   - 시각화: {exp_dir}/plots/ (덮어쓰기)")
    print(f"   - 리포트: {exp_dir}/reports/ (덮어쓰기)")
    
    return exp_dir


# ==============================
# CNN 모델만 훈련
# ==============================
def train_cnn_models():
    """CNN 모델들 비교 훈련"""
    print("=" * 60)
    print(" CNN 모델 훈련 시작")
    print("=" * 60)

    # 실험 폴더 생성
    exp_dir = create_experiment_dir()

    # 데이터 로드
    data_loader = DataLoader()
    train_data, val_data, class_names = data_loader.load_data()

    # CNN 모델 생성
    cnn_models = CNNModels()
    models_dict = cnn_models.get_all_models()

    # 훈련 (exp_dir을 모델 저장 경로로 전달)
    trainer = ModelTrainer()
    histories = trainer.train_multiple_models(models_dict, train_data, val_data)

    # 성능 비교
    results = trainer.compare_models()

    # 시각화 (plots 폴더에 저장)
    visualizer = Visualizer(save_dir=os.path.join(exp_dir, 'plots'))
    visualizer.generate_all_plots(histories, results)

    return histories, results, class_names, exp_dir


# ==============================
# CNN + Transfer Learning 최종 비교
# ==============================
def train_final_models():
    """최종 모델 비교 (CNN + Transfer Learning)"""
    print("=" * 60)
    print(" 최종 모델 비교 훈련 시작")
    print("=" * 60)

    # 실험 폴더 생성
    exp_dir = create_experiment_dir()

    # 데이터 로드
    data_loader = DataLoader()
    train_data, val_data, class_names = data_loader.load_data()

    # 모델 생성
    cnn_models = CNNModels()
    transfer_models = TransferModels()

    final_models = {
        "CNN_With_Dropout": cnn_models.build_dropout(),
        **transfer_models.get_all_models()
    }

    # 훈련
    trainer = ModelTrainer()
    histories = trainer.train_multiple_models(final_models, train_data, val_data)

    # 성능 비교
    results = trainer.compare_models()

    # 시각화
    visualizer = Visualizer(save_dir=os.path.join(exp_dir, 'plots'))
    visualizer.generate_all_plots(histories, results)

    return histories, results, class_names, exp_dir


# ==============================
# Groq LLM 분석
# ==============================
def analyze_with_groq_llm(results, histories, exp_dir):
    """Groq LLM을 사용한 분석 및 리포트 생성"""
    if not GROQ_AVAILABLE:
        print("❌ Groq 패키지가 설치되지 않았습니다.")
        print("   설치: pip install groq")
        return
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   https://console.groq.com 에서 무료 API 키를 발급받으세요.")
        return
    
    print("\n" + "=" * 60)
    print(" Groq LLM 분석 시작")
    print("=" * 60)
    
    analyzer = GroqLLMAnalyzer()
    reports_dir = os.path.join(exp_dir, 'reports')
    
    # 1. 모델 성능 분석
    print("\n📊 모델 성능 분석 중 (Groq LLM)...")
    try:
        analysis = analyzer.analyze_model_performance(
            results, 
            model_choice="llama3-70b"  # llama-3.3-70b-versatile
        )
        print("\n" + analysis)
        
        # 저장
        with open(os.path.join(reports_dir, 'performance_analysis.md'), 'w', encoding='utf-8') as f:
            f.write("# 모델 성능 분석 (Groq LLM)\n\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(analysis)
        print(f"✓ 성능 분석 저장: {reports_dir}/performance_analysis.md")
        
    except Exception as e:
        print(f"❌ 성능 분석 실패: {e}")
    
    # 2. 훈련 리포트 생성
    if histories:
        print("\n📝 훈련 리포트 생성 중 (Groq LLM)...")
        try:
            report = analyzer.generate_training_report(
                histories,
                model_choice="llama3-70b"
            )
            print("\n" + report)
            
            # 저장
            with open(os.path.join(reports_dir, 'training_report.md'), 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 훈련 리포트 저장: {reports_dir}/training_report.md")
            
        except Exception as e:
            print(f"❌ 훈련 리포트 생성 실패: {e}")
    
    return analysis

# ==============================
# main
# ==============================
def main():
    parser = argparse.ArgumentParser(description="차량 이미지 분류 프로젝트")
    parser.add_argument(
        "--mode",
        type=str,
        default="cnn",
        choices=["cnn", "final"],
        help="cnn: CNN 모델 비교 / final: CNN + Transfer Learning 비교",
    )
    parser.add_argument(
        "--use-groq",
        action="store_true",
        help="Groq LLM 분석 사용 여부",
    )

    args = parser.parse_args()

    if args.mode == "cnn":
        histories, results, class_names, exp_dir = train_cnn_models()

    elif args.mode == "final":
        histories, results, class_names, exp_dir = train_final_models()

    # Groq LLM 분석
    if args.use_groq:
        analyze_with_groq_llm(results, histories, exp_dir)

    print("\n" + "=" * 60)
    print(" 프로그램 실행 완료!")
    print(f" 결과 위치: {exp_dir}")
    print("=" * 60)
    print(f"\n📂 결과 확인:")
    print(f"   - 모델: {exp_dir}/models/")
    print(f"   - 시각화: {exp_dir}/plots/")
    print(f"   - 리포트: {exp_dir}/reports/")
    print(f"\n💡 다음 실행 시 자동으로 덮어쓰기됩니다.")


if __name__ == "__main__":
    main()