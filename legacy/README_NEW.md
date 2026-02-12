# RAG 실험 프레임워크 (분리 버전)

RAG 답변 생성과 평가를 독립적인 두 파이프라인으로 분리했습니다.

## 폴더 구조

```
Exp_2/
├── 1_rag_generation/          # RAG 답변 생성 파이프라인
│   ├── config.py              # 생성 설정
│   ├── utils.py
│   ├── chunking_strategies.py # 청킹 전략 클래스들
│   ├── build_pdf_chroma.py    # Vector DB 구축
│   ├── rag_answer_pipeline.py # RAG 답변 생성
│   ├── run.py                 # 전체 파이프라인 실행
│   ├── README.md
│   └── output/                # 생성된 답변 저장
│       └── {experiment_name}/
│           └── rag_answers.jsonl
│
└── 2_evaluation/              # RAGAS 평가 파이프라인
    ├── config.py              # 평가 설정
    ├── utils.py
    ├── ragas_eval.py          # RAGAS 메트릭 평가
    ├── evaluate_accuracy.py   # LLM 기반 정확도 평가
    ├── run.py                 # 전체 평가 실행
    ├── README.md
    ├── input/                 # 평가할 답변 파일
    │   └── {experiment_name}/
    │       └── rag_answers.jsonl
    └── output/                # 평가 결과 저장
        └── {experiment_name}/
            ├── ragas_results_*.csv
            └── evaluation_log_*.md
```

## 사용 흐름

### 1단계: RAG 답변 생성

```bash
cd 1_rag_generation

# config.py 설정 변경
# CURRENT_CONFIG = CONFIG_PARENT_CHILD

python run.py
```

**출력**: `output/parent_child/rag_answers.jsonl`

### 2단계: 답변 파일 복사

생성된 답변을 평가 폴더로 복사:

```bash
# Windows PowerShell
Copy-Item -Recurse "1_rag_generation\output\parent_child" -Destination "2_evaluation\input\"
```

### 3단계: RAGAS 평가

```bash
cd 2_evaluation

# config.py 설정 변경
# experiment_name="parent_child"

python run.py
```

**출력**: 
- `output/parent_child/ragas_results_*.csv`
- `output/parent_child/evaluation_log_*.md`

## 장점

### ✅ 독립성
- 각 파이프라인이 완전히 독립적으로 실행 가능
- 답변 생성 없이 평가만 반복 가능
- 다른 시스템에서 생성한 답변도 평가 가능

### ✅ 유연성
- 답변 생성 설정과 평가 설정을 별도로 관리
- 한 번 생성한 답변으로 다양한 평가 실험 가능

### ✅ 효율성
- 평가만 필요할 때 답변 재생성 불필요
- 여러 실험 결과를 한 곳에서 비교 평가 가능

## 빠른 시작

### RAG 답변 생성
```bash
cd 1_rag_generation
python run.py
```

### 평가 (답변 파일 복사 후)
```bash
cd 2_evaluation
python run.py
```

## 상세 문서

- [1_rag_generation/README.md](1_rag_generation/README.md)
- [2_evaluation/README.md](2_evaluation/README.md)

## 주의사항

1. **답변 파일 복사**: 생성 후 수동으로 평가 폴더에 복사 필요
2. **실험 이름 일치**: 두 config.py의 `experiment_name`이 일치해야 함
3. **환경 변수**: 각 폴더에 `.env` 파일 필요 (OPENAI_API_KEY)
