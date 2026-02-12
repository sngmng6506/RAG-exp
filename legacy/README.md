# RAG 실험 프레임워크

청킹 전략, Retrieval 파라미터 등을 변경하며 RAG 성능을 실험하는 프레임워크입니다.

## 폴더 구조

```
Exp_2/
├── config.py                  # 실험 설정 (여기만 수정!)
├── utils.py                   # 공통 유틸리티 함수
├── chunking_strategies.py     # 청킹 전략 클래스들 (Strategy Pattern)
├── run_experiment.py          # 전체 파이프라인 실행
├── build_pdf_chroma.py        # 1단계: PDF → Vector DB
├── rag_answer_pipeline.py     # 2단계: RAG 답변 생성
├── ragas_eval.py              # 3단계: RAGAS 평가
├── evaluate_accuracy.py       # 4단계: LLM 기반 정확도 평가
└── log/
    └── {experiment_name}/     # 실험별 결과 저장
        ├── rag_answers.jsonl
        ├── ragas_checkpoint.json
        ├── ragas_results_*.csv
        └── evaluation_log_*.md
```

## 빠른 시작

### 전체 파이프라인 실행 (권장)
```bash
python run_experiment.py
```

이 명령어 하나로 다음 단계가 모두 실행됩니다:
1. Vector DB 구축
2. RAG 답변 생성
3. RAGAS 평가
4. LLM 기반 정확도 평가

### 옵션
```bash
# Vector DB가 이미 있으면 구축 건너뛰기
python run_experiment.py --skip-build

# RAGAS 평가만 실행
python run_experiment.py --only-ragas

# 정확도 평가만 실행
python run_experiment.py --only-accuracy
```

## 상세 사용법

### 1. 실험 설정

`config.py`에서 `CURRENT_CONFIG`만 변경하면 됩니다:

```python
# config.py 맨 아래

# 기본 실험 실행
CURRENT_CONFIG = CONFIG_BASELINE

# 또는 큰 청크 실험
CURRENT_CONFIG = CONFIG_LARGE_CHUNK

# 또는 새로운 실험 설정
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="my_experiment",
    collection_name="pdf_rag_my_exp",
    chunk_size=800,
    chunk_overlap=80,
    retriever_top_k=15,
    rerank_top_k=7,
)
```

### 2. 실험 실행

#### 방법 1: 전체 파이프라인 (권장)
```bash
python run_experiment.py
```

#### 방법 2: 단계별 실행
```bash
# (1) Vector DB 구축
python build_pdf_chroma.py

# (2) RAG 답변 생성
python rag_answer_pipeline.py

# (3) RAGAS 평가
python ragas_eval.py

# (4) LLM 기반 정확도 평가
python evaluate_accuracy.py
```

각 단계의 역할:
- **build_pdf_chroma.py**: PDF를 청킹하여 Chroma DB에 저장
- **rag_answer_pipeline.py**: 질문에 대한 RAG 답변 생성
- **ragas_eval.py**: RAGAS 메트릭 평가 (체크포인트 지원)
- **evaluate_accuracy.py**: LLM이 생성 답변과 정답을 비교하여 정확도 계산

## 주요 설정 항목

### ExperimentConfig

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `experiment_name` | 실험 이름 (log 폴더명) | `"baseline"` |
| `collection_name` | Vector DB collection 이름 | `"pdf_rag"` |
| `chunk_size` | 청크 크기 | `500` |
| `chunk_overlap` | 청크 오버랩 | `50` |
| `retriever_top_k` | 초기 검색 문서 수 | `10` |
| `rerank_top_k` | 재순위 후 문서 수 | `5` |
| `max_new_tokens` | 생성 최대 토큰 수 | `512` |
| `eval_model` | RAGAS 평가 모델 | `"gpt-4o-mini"` |

## 실험 예시

### 청킹 전략 비교

#### 일반 청킹 크기 비교
```python
# config.py

# 실험 1: 작은 청크
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="chunk_300",
    collection_name="pdf_rag_300",
    chunk_size=300,
    chunk_overlap=30,
)

# 실험 2: 중간 청크
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="chunk_500",
    collection_name="pdf_rag_500",
    chunk_size=500,
    chunk_overlap=50,
)

# 실험 3: 큰 청크
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="chunk_1000",
    collection_name="pdf_rag_1000",
    chunk_size=1000,
    chunk_overlap=100,
)
```

#### Parent-Child Chunking
작은 청크(child)로 정확하게 검색하고, 큰 청크(parent)로 충분한 맥락 제공:

```python
# config.py

CURRENT_CONFIG = ExperimentConfig(
    experiment_name="parent_child",
    collection_name="pdf_rag_parent_child",
    use_parent_child=True,
    parent_chunk_size=2000,    # 반환용 (큰 맥락)
    parent_chunk_overlap=200,
    child_chunk_size=400,      # 검색용 (정확한 매칭)
    child_chunk_overlap=50,
)
```

각 실험마다 3단계를 순차 실행하면 됩니다.

### Retrieval 파라미터 비교
```python
# collection_name은 동일 (같은 Vector DB 사용)

# 실험 1: 적은 문서
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="retrieval_5",
    collection_name="pdf_rag_baseline",  # 동일
    retriever_top_k=10,
    rerank_top_k=3,
)

# 실험 2: 많은 문서
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="retrieval_10",
    collection_name="pdf_rag_baseline",  # 동일
    retriever_top_k=20,
    rerank_top_k=10,
)
```

## 결과 확인

### Log 폴더 구조
```
log/
├── baseline/
│   ├── rag_answers.jsonl
│   ├── ragas_results_20260205_111120.csv
│   └── evaluation_log_20260205_112617.md
├── large_chunk/
│   ├── rag_answers.jsonl
│   ├── ragas_results_*.csv
│   └── evaluation_log_*.md
└── small_chunk/
    ├── rag_answers.jsonl
    ├── ragas_results_*.csv
    └── evaluation_log_*.md
```

### 평가 지표

#### RAGAS 메트릭 (`ragas_results_*.csv`)
- **faithfulness**: 답변의 사실성 (문맥 기반)
- **answer_relevancy**: 답변의 관련성
- **context_precision**: 검색된 문맥의 정확도
- **context_recall**: 필요한 문맥의 재현율

#### LLM 기반 정확도 (`evaluation_log_*.md`)
- **Accuracy**: 생성 답변이 정답과 일치하는 비율
- **샘플별 상세 판단**: 각 질문에 대한 정답/오답 판단 근거

## 주요 특징

- ✅ **원클릭 실행**: `run_experiment.py` 하나로 전체 파이프라인 실행
- ✅ **체크포인트**: RAGAS 평가 중단 시 자동 재개
- ✅ **자동 환경 설정**: `.env` 파일 자동 로드 및 검증
- ✅ **상세 로그**: 마크다운 형식의 평가 결과 리포트
- ✅ **유연한 설정**: `config.py`만 수정하여 다양한 실험 가능
- ✅ **확장 가능한 아키텍처**: Strategy Pattern으로 청킹 전략 쉽게 추가

## 주의사항

1. **청킹 전략 변경 시**: `collection_name`을 다르게 설정
2. **Retrieval 파라미터만 변경 시**: `collection_name` 동일하게 유지
3. **.env 파일**: `OPENAI_API_KEY` 필수
4. **체크포인트**: `ragas_eval.py` 중단 시 자동 저장되어 재실행하면 이어서 진행

## 트러블슈팅

### Vector DB에 같은 collection이 있어요
```python
# config.py에서 collection_name 변경
collection_name="pdf_rag_v2"
```

### RAGAS 평가가 중단되었어요
```bash
# 체크포인트부터 자동 재개됩니다
python run_experiment.py --only-ragas
```

### 여러 실험 결과를 비교하고 싶어요
```python
import pandas as pd

# RAGAS 메트릭 비교
df1 = pd.read_csv("log/baseline/ragas_results_*.csv")
df2 = pd.read_csv("log/large_chunk/ragas_results_*.csv")

print("Baseline:", df1[["faithfulness", "answer_relevancy"]].mean())
print("Large Chunk:", df2[["faithfulness", "answer_relevancy"]].mean())
```

### 평가 로그를 보고 싶어요
```bash
# 마크다운 파일로 자동 생성됩니다
cat log/baseline/evaluation_log_*.md
```

### 새로운 청킹 전략을 추가하고 싶어요
`chunking_strategies.py`에 새 클래스를 추가하면 됩니다:

```python
# chunking_strategies.py

class MyCustomChunkingStrategy(BaseChunkingStrategy):
    """커스텀 청킹 전략"""
    
    def build_vectorstore(self, documents, embeddings):
        # 구축 로직 구현
        pass
    
    def get_retriever(self, embeddings):
        # Retriever 로드 로직 구현
        pass
    
    def get_strategy_info(self) -> dict:
        # 전략 정보 반환
        return {"type": "My Custom Chunking"}

# config.py에서 사용
CURRENT_CONFIG = ExperimentConfig(
    experiment_name="my_custom",
    collection_name="pdf_rag_custom",
    # 커스텀 파라미터 추가
)
```
