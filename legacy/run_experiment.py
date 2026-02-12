"""
실험 전체 파이프라인을 순차 실행하는 스크립트

사용법:
    python run_experiment.py                  # 전체 실행
    python run_experiment.py --skip-build     # Vector DB 구축 건너뛰기
    python run_experiment.py --only-ragas     # RAGAS 평가만 실행
    python run_experiment.py --only-accuracy  # 정확도 평가만 실행
"""
import argparse
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
    parser = argparse.ArgumentParser(description="RAG 실험 파이프라인 실행")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Vector DB 구축 단계 건너뛰기",
    )
    parser.add_argument(
        "--only-ragas",
        action="store_true",
        help="RAGAS 평가만 실행",
    )
    parser.add_argument(
        "--only-accuracy",
        action="store_true",
        help="정확도 평가만 실행",
    )
    args = parser.parse_args()

    cfg = CURRENT_CONFIG

    print_section(f"🚀 실험 시작: {cfg.experiment_name}")
    print(f"Collection: {cfg.collection_name}")
    print(f"Chunk 설정: size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")
    print(f"Retrieval: top_k={cfg.retriever_top_k}, rerank={cfg.rerank_top_k}")
    print(f"Log 저장: {cfg.log_dir}")

    if args.only_ragas:
        print_section("3️⃣ RAGAS 평가")
        run_command("ragas_eval.py")
    elif args.only_accuracy:
        print_section("4️⃣ 정확도 평가")
        run_command("evaluate_accuracy.py")
    else:
        # 전체 파이프라인 실행
        if not args.skip_build:
            print_section("1️⃣ Vector DB 구축")
            run_command("build_pdf_chroma.py")

        print_section("2️⃣ RAG 답변 생성")
        run_command("rag_answer_pipeline.py")

        print_section("3️⃣ RAGAS 평가")
        run_command("ragas_eval.py")
        
        print_section("4️⃣ 정확도 평가")
        run_command("evaluate_accuracy.py")

    print_section("✅ 모든 작업 완료!")
    print(f"결과 확인: {cfg.log_dir}")


if __name__ == "__main__":
    main()
