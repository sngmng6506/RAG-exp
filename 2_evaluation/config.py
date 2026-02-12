"""
RAGAS 평가 설정 파일
"""
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """RAGAS 평가 설정"""
    
    # ========== 실험 식별 ==========
    experiment_name: str = "baseline"
    
    # ========== 평가 설정 ==========
    eval_model: str = "gpt-4o-mini"  # RAGAS 평가용 모델
    
    # ========== 필터링 옵션 ==========
    filter_by_incorrect: bool = False  # True면 incorrect_indices.json의 인덱스만 처리
    
    # ========== 자동 설정 ==========
    _exp_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    _llm_exp_root: Path = field(init=False)
    
    def __post_init__(self):
        self._llm_exp_root = self._exp_root.parent.parent
    
    @property
    def input_dir(self) -> Path:
        """평가할 답변 파일 디렉토리"""
        return self._exp_root / "input" / self.experiment_name
    
    @property
    def output_dir(self) -> Path:
        """평가 결과 출력 디렉토리"""
        return self._exp_root / "output" / self.experiment_name
    
    @property
    def rag_answers_path(self) -> Path:
        """RAG 답변 JSONL 경로"""
        return self.input_dir / "rag_answers.jsonl"
    
    @property
    def checkpoint_path(self) -> Path:
        """체크포인트 파일 경로"""
        return self.output_dir / "ragas_checkpoint.json"
    
    @property
    def log_dir(self) -> Path:
        """호환성을 위한 별칭 (output_dir과 동일)"""
        return self.output_dir
    
    @property
    def embedding_model_path(self) -> Path:
        """임베딩 모델 경로"""
        return self._llm_exp_root / "models" / "embedding" / "ko-sbert-sts"
    
    @property
    def incorrect_indices_path(self) -> Path:
        """틀린 문항 인덱스 JSON 경로 (Exp_2 루트의 공통 파일)"""
        return self._exp_root.parent / "incorrect_indices.json"


# ========================================
# 현재 설정 (여기만 변경하면 됨!)
# ========================================
# 전체 평가
CONFIG_FULL = EvalConfig(
    experiment_name="parent_child",
    eval_model="gpt-5.2",
)

# 틀린 문항만 평가
CONFIG_FILTERED = EvalConfig(
    experiment_name="parent_child",
    eval_model="gpt-5.2",
    filter_by_incorrect=True,
)

# 현재 설정
CURRENT_CONFIG = CONFIG_FULL


if __name__ == "__main__":
    cfg = CURRENT_CONFIG
    print(f"실험 이름: {cfg.experiment_name}")
    print(f"입력 디렉토리: {cfg.input_dir}")
    print(f"출력 디렉토리: {cfg.output_dir}")
    print(f"평가 모델: {cfg.eval_model}")
    print(f"RAG 답변 파일: {cfg.rag_answers_path}")
