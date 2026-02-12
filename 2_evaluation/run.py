"""
RAGAS 평가 파이프라인 실행
"""
import subprocess
import sys
from pathlib import Path

from config import CURRENT_CONFIG


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_command(script_name):
    """Python 스크립트 실행"""
    print(f"▶ {script_name} 실행 중...\n")
    result = subprocess.run([sys.executable, script_name], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n❌ {script_name} 실행 실패 (exit code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n✅ {script_name} 완료")


def main():
    cfg = CURRENT_CONFIG
    
    print_section(f"📊 RAGAS 평가 시작: {cfg.experiment_name}")
    print(f"평가 모델: {cfg.eval_model}")
    print(f"입력 경로: {cfg.input_dir}")
    print(f"출력 경로: {cfg.output_dir}")
    
    # RAG 답변 파일 확인
    if not cfg.rag_answers_path.exists():
        print(f"\n❌ RAG 답변 파일이 없습니다: {cfg.rag_answers_path}")
        print(f"\n먼저 1_rag_generation에서 답변을 생성하고,")
        print(f"output 폴더를 이 폴더의 input/{cfg.experiment_name}로 복사하세요.")
        sys.exit(1)
    
    # 1단계: RAGAS 평가
    print_section("1️⃣ RAGAS 메트릭 평가")
    run_command("ragas_eval.py")
    
    # 2단계: LLM 기반 정확도 평가
    print_section("2️⃣ LLM 기반 정확도 평가")
    run_command("evaluate_accuracy.py")
    
    print_section("✅ 평가 완료!")
    print(f"결과 확인: {cfg.output_dir}")


if __name__ == "__main__":
    main()
