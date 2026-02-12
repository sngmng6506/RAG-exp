# RAGAS 평가 파이프라인

RAG 답변 파일(`rag_answers.jsonl`)을 평가합니다.

## 설치

Exp_2 공통 의존성 설치:

```bash
# Exp_2 폴더 기준
pip install -r requirements.txt
```

## 사용법

### 1. 답변 파일 준비

`1_rag_generation`에서 생성한 답변 파일을 복사:

```bash
# Windows PowerShell
Copy-Item -Recurse "..\1_rag_generation\output\{experiment_name}" -Destination "input\"

# 예시: parent_child 실험
Copy-Item -Recurse "..\1_rag_generation\output\parent_child" -Destination "input\"
```

폴더 구조:
```
2_evaluation/
├── input/
│   └── parent_child/
│       └── rag_answers.jsonl
└── output/
    └── parent_child/
        ├── ragas_results_*.csv
        └── evaluation_log_*.md
```

### 2. 설정 변경

`config.py`에서 실험 이름 설정:

```python
CURRENT_CONFIG = EvalConfig(
    experiment_name="parent_child",  # input 폴더 내 폴더명과 일치
    eval_model="gpt-5.2",
)
```

### 3. 실행

```bash
python run.py
```

또는 단계별 실행:

```bash
# 1단계: RAGAS 평가
python ragas_eval.py

# 2단계: 정확도 평가
python evaluate_accuracy.py
```

## 출력

- `output/{experiment_name}/ragas_results_*.csv`: RAGAS 메트릭 결과
- `output/{experiment_name}/evaluation_log_*.md`: 정확도 평가 상세 로그

## 파일 구조

- `config.py`: 평가 설정
- `utils.py`: 공통 유틸리티 함수
- `ragas_eval.py`: RAGAS 메트릭 평가
- `evaluate_accuracy.py`: LLM 기반 정확도 평가
- `run.py`: 전체 평가 파이프라인 실행
