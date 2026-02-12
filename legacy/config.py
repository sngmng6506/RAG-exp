"""
실험 설정 파일
각 실험마다 experiment_name, collection_name 등을 변경하여 사용
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """실험 설정"""
    
    # ========== 실험 식별 ==========
    experiment_name: str = "parent-child-chunking"  # 실험 이름 (log 폴더명에 사용)
    
    # ========== Vector DB 설정 ==========
    collection_name: str = "pdf_rag_parent_child"  # Chroma collection 이름 (청킹 전략별로 다른 이름 사용)
    
    # ========== Chunking 설정 ==========
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # ========== Parent-Child Chunking 설정 ==========
    use_parent_child: bool = False  # Parent-Child 전략 사용 여부
    parent_chunk_size: int = 2000   # Parent 청크 크기 (반환용, 큰 맥락)
    parent_chunk_overlap: int = 200 # Parent 청크 오버랩
    child_chunk_size: int = 400     # Child 청크 크기 (검색용, 작은 청크)
    child_chunk_overlap: int = 50   # Child 청크 오버랩
    
    # ========== Retrieval 설정 ==========
    retriever_top_k: int = 10  # 초기 검색 문서 수
    rerank_top_k: int = 5      # 재순위 후 최종 문서 수
    
    # ========== Generation 설정 ==========
    generate_answers: bool = True  # False면 ground_truth만 사용
    max_new_tokens: int = 512
    
    # ========== 평가 설정 ==========
    eval_model: str = "gpt-5.2"  # RAGAS 평가용 모델 #"gpt-4o-mini"
    
    # ========== 자동 설정 (수정 불필요) ==========
    _exp_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    _llm_exp_root: Path = field(init=False)
    
    def __post_init__(self):
        self._llm_exp_root = self._exp_root.parent
    
    @property
    def log_dir(self) -> Path:
        """실험별 log 디렉토리"""
        return self._exp_root / "log" / self.experiment_name
    
    @property
    def chroma_dir(self) -> Path:
        """Vector DB 저장 경로"""
        return self._llm_exp_root / "storage" / "vector_db" / "chroma"
    
    @property
    def pdf_dir(self) -> Path:
        """PDF 소스 경로"""
        return self._llm_exp_root / "data" / "raw" / "pdf"
    
    @property
    def xlsx_path(self) -> Path:
        """평가 데이터셋 경로"""
        return self._llm_exp_root / "evaluation" / "ragas" / "datasets" / "qac_dataset_105_v1.xlsx"
    
    @property
    def embedding_model_path(self) -> Path:
        """임베딩 모델 경로"""
        return self._llm_exp_root / "models" / "embedding" / "ko-sbert-sts"
    
    @property
    def llm_path(self) -> Path:
        """생성 모델 경로"""
        return self._llm_exp_root / "models" / "llm" / "gemma3-12b-it"
    
    @property
    def reranker_model_path(self) -> Path:
        """재순위 모델 경로"""
        return self._llm_exp_root / "models" / "reranker" / "bge-reranker-v2-m3"
    
    @property
    def checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.log_dir / "ragas_checkpoint.json"
    
    @property
    def rag_answers_path(self) -> Path:
        """RAG 답변 JSONL 경로"""
        return self.log_dir / "rag_answers.jsonl"
    
    @property
    def incorrect_indices_path(self) -> Path:
        """틀린 문항 인덱스 JSON 경로"""
        return self.log_dir / "incorrect_indices.json"
    
    @property
    def docstore_dir(self) -> Path:
        """Parent-Child Chunking의 docstore 경로"""
        return self.chroma_dir / f"{self.collection_name}_docstore"
    
    @property
    def docstore_path(self) -> Path:
        """Parent-Child Chunking의 docstore 파일 경로"""
        return self.docstore_dir


# ========================================
# 실험별 설정 예시
# ========================================

# 기본 실험
CONFIG_BASELINE = ExperimentConfig(
    experiment_name="baseline",
    collection_name="pdf_rag_baseline",
    chunk_size=500,
    chunk_overlap=50,
)

# Retrieval 실험
CONFIG_HIGH_RETRIEVAL = ExperimentConfig(
    experiment_name="high_retrieval",
    collection_name="pdf_rag_baseline",
    retriever_top_k=10,
    rerank_top_k=5,
)

# Parent-Child Chunking 실험
CONFIG_PARENT_CHILD = ExperimentConfig(
    experiment_name="parent_child",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
    retriever_top_k=10,
    rerank_top_k=5,
)


# ========================================
# 현재 실험 설정 (여기만 변경하면 됨!)
# ========================================
CURRENT_CONFIG = CONFIG_PARENT_CHILD  # Parent-Child Chunking 실험
# CURRENT_CONFIG = CONFIG_BASELINE


if __name__ == "__main__":
    # Config 확인용
    cfg = CURRENT_CONFIG
    print(f"실험 이름: {cfg.experiment_name}")
    print(f"Collection: {cfg.collection_name}")
    print(f"Log 디렉토리: {cfg.log_dir}")
    
    if cfg.use_parent_child:
        print(f"📚 Parent-Child Chunking")
        print(f"  Parent: size={cfg.parent_chunk_size}, overlap={cfg.parent_chunk_overlap}")
        print(f"  Child: size={cfg.child_chunk_size}, overlap={cfg.child_chunk_overlap}")
        print(f"  Docstore: {cfg.docstore_path}")
    else:
        print(f"📄 일반 Chunking")
        print(f"  Chunk: size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")
    
    print(f"Retrieval 설정: top_k={cfg.retriever_top_k}, rerank={cfg.rerank_top_k}")
