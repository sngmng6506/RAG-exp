"""
RAG 답변 생성 파이프라인 실행
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
    print(f">> {script_name} 실행 중...\n")
    result = subprocess.run([sys.executable, script_name], cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n[FAIL] {script_name} 실행 실패 (exit code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[OK] {script_name} 완료")


def main():
    cfg = CURRENT_CONFIG
    
    print_section(f"RAG 답변 생성 시작: {cfg.experiment_name}")
    print(f"Collection: {cfg.collection_name}")
    print(f"출력 경로: {cfg.output_dir}")
    
    if cfg.use_parent_child:
        print(f"청킹 전략: Parent-Child")
        print(f"  Parent: size={cfg.parent_chunk_size}, overlap={cfg.parent_chunk_overlap}")
        print(f"  Child: size={cfg.child_chunk_size}, overlap={cfg.child_chunk_overlap}")
    else:
        print(f"청킹 전략: Simple")
        print(f"  Chunk: size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")
    
    print(f"Retrieval: top_k={cfg.retriever_top_k}, rerank={cfg.rerank_top_k}")
    
    # 1단계: Vector DB 구축
    print_section("[1/3] Vector DB 구축")
    run_command("build_pdf_chroma.py")

    # 2단계: BM25 인덱스 구축 (Hybrid 모드)
    if getattr(cfg, "use_hybrid_retriever", False):
        print_section("[2/3] BM25 인덱스 구축")
        run_command("build_bm25_index.py")

    # 3단계: RAG 답변 생성
    print_section("[3/3] RAG 답변 생성")
    run_command("rag_answer_pipeline.py")
    
    print_section("[OK] RAG 답변 생성 완료!")
    print(f"출력 파일: {cfg.rag_answers_path}")
    print(f"\n다음 단계: 2_evaluation 폴더로 이동하여 평가 실행")


if __name__ == "__main__":
    main()
