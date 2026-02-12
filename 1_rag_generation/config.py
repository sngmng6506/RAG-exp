"""
RAG 답변 생성 설정 파일
"""
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """RAG 답변 생성 설정"""
    
    # ========== 실험 식별 ==========
    experiment_name: str = "baseline"
    
    # ========== Vector DB 설정 ==========
    collection_name: str = "pdf_rag_baseline"
    
    # ========== Chunking 설정 ==========
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # ========== 필터링 옵션 ==========
    filter_by_incorrect: bool = False  # True면 incorrect_indices.json의 인덱스만 처리
    
    # ========== Parent-Child Chunking 설정 ==========
    use_parent_child: bool = False
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    
    # ========== Retrieval 설정 ==========
    retriever_top_k: int = 10
    rerank_top_k: int = 5
    
    # ========== Hybrid Retrieval (BM25 + Vector) ==========
    use_hybrid_retriever: bool = False
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    
    # ========== 빌드 스킵 옵션 ==========
    skip_vector_if_exists: bool = True
    skip_bm25_if_exists: bool = True
    
    # ========== Generation 설정 ==========
    generate_answers: bool = True
    max_new_tokens: int = 512
    
    # ========== 자동 설정 ==========
    _exp_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    _llm_exp_root: Path = field(init=False)
    
    def __post_init__(self):
        self._llm_exp_root = self._exp_root.parent.parent
    
    @property
    def output_dir(self) -> Path:
        """답변 출력 디렉토리"""
        return self._exp_root / "output" / self.experiment_name
    
    @property
    def chroma_dir(self) -> Path:
        """Vector DB 저장 경로"""
        return self._llm_exp_root / "storage" / "vector_db" / "chroma"
    
    @property
    def bm25_dir(self) -> Path:
        """BM25 인덱스 저장 경로"""
        return self._llm_exp_root / "storage" / "bm25"
    
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
    def rag_answers_path(self) -> Path:
        """RAG 답변 JSONL 경로"""
        return self.output_dir / "rag_answers.jsonl"
    
    @property
    def incorrect_indices_path(self) -> Path:
        """틀린 문항 인덱스 JSON 경로 (Exp_2 루트의 공통 파일)"""
        return self._exp_root.parent / "incorrect_indices.json"
    
    @property
    def bm25_index_path(self) -> Path:
        """BM25 인덱스 파일 경로"""
        return self.bm25_dir / f"{self.collection_name}_bm25.pkl"
    
    @property
    def docstore_path(self) -> Path:
        """Parent-Child Chunking의 docstore 경로"""
        return self.chroma_dir / f"{self.collection_name}_docstore"


# ========================================
# 실험 설정 예시
# ========================================

# 기본 실험
CONFIG_BASELINE = RAGConfig(
    experiment_name="baseline",
    collection_name="pdf_rag_baseline",
    chunk_size=500,
    chunk_overlap=50,
)

# Parent-Child Chunking (Baseline)
CONFIG_PARENT_CHILD_BASELINE = RAGConfig(
    experiment_name="parent_child_baseline",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
    use_hybrid_retriever=False,
)

# Parent-Child Chunking (Hybrid BM25 + Vector)
CONFIG_PARENT_CHILD_HYBRID = RAGConfig(
    experiment_name="parent_child_hybrid",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
    use_hybrid_retriever=True,
)

# Parent-Child + 틀린 문항만 필터링 (Baseline)
CONFIG_PARENT_CHILD_FILTERED = RAGConfig(
    experiment_name="parent_child_baseline",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
    filter_by_incorrect=True,  # 틀린 문항만 처리
    use_hybrid_retriever=False,
)

# Parent-Child + 틀린 문항만 필터링 (Hybrid BM25 + Vector)
CONFIG_PARENT_CHILD_HYBRID_FILTERED = RAGConfig(
    experiment_name="parent_child_hybrid",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
    filter_by_incorrect=True,  # 틀린 문항만 처리
    use_hybrid_retriever=True,
)


# ========================================
# 현재 설정 (여기만 변경하면 됨!)
# ========================================
CURRENT_CONFIG = CONFIG_PARENT_CHILD_HYBRID_FILTERED


if __name__ == "__main__":
    cfg = CURRENT_CONFIG
    print(f"실험 이름: {cfg.experiment_name}")
    print(f"Collection: {cfg.collection_name}")
    print(f"출력 디렉토리: {cfg.output_dir}")
    
    if cfg.use_parent_child:
        print(f"[Parent-Child Chunking]")
        print(f"  Parent: size={cfg.parent_chunk_size}, overlap={cfg.parent_chunk_overlap}")
        print(f"  Child: size={cfg.child_chunk_size}, overlap={cfg.child_chunk_overlap}")
    else:
        print(f"[Simple Chunking]")
        print(f"  Chunk: size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")
    
    print(f"Retrieval: top_k={cfg.retriever_top_k}, rerank={cfg.rerank_top_k}")
