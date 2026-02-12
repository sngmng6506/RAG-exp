# RAG 답변 생성 파이프라인

청킹 전략, Retrieval 파라미터 등을 설정하여 RAG 답변을 생성합니다.

## 설치

Exp_2 공통 의존성 설치:

```bash
# Exp_2 폴더 기준
pip install -r requirements.txt
```

## 사용법

### 1. 설정 변경

`config.py`에서 `CURRENT_CONFIG` 변경:

```python
# 일반 청킹
CURRENT_CONFIG = CONFIG_BASELINE

# 또는 Parent-Child 청킹
CURRENT_CONFIG = CONFIG_PARENT_CHILD
```

### 2. 실행

```bash
python run.py
```

또는 단계별 실행:

```bash
# 1단계: Vector DB 구축
python build_pdf_chroma.py

# 2단계: BM25 인덱스 구축 (Hybrid 모드일 때)
python build_bm25_index.py

# 3단계: RAG 답변 생성
python rag_answer_pipeline.py
```

## 출력

- `output/{experiment_name}/rag_answers.jsonl`: RAG 답변 파일
- 이 파일을 `2_evaluation` 폴더로 복사하여 평가 진행

## 파일 구조

- `config.py`: 실험 설정
- `utils.py`: 공통 유틸리티 함수
- `chunking_strategies.py`: 청킹 전략 클래스들
- `build_pdf_chroma.py`: Vector DB 구축
- `rag_answer_pipeline.py`: RAG 답변 생성
- `run.py`: 전체 파이프라인 실행
