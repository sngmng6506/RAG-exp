# 4_fix_incorrect_indices

`incorrect_indices.json`의 인덱스를 subset 기준에서 전체 기준으로 변환하는 스크립트입니다.

## 배경

1. 원래 105개 전체 문항에서 baseline RAG로 평가
2. 그 중 일부(38개)가 오답으로 판별되어 subset으로 분리
3. subset(38개)을 Parent-Child RAG로 재평가
4. 재평가 결과 중 일부가 여전히 오답 (29개)

이때 `incorrect_indices.json`에는 subset 내부의 인덱스(0~37)가 저장되어 있어서,
전체 105개 기준의 인덱스로 변환이 필요합니다.

## 사용법

```bash
cd 4_fix_incorrect_indices
python fix_incorrect_indices.py
```

## 입력 파일

- `3_re_evaluation/results/total/ragas_results_*.csv`: 105개 전체 결과
- `3_re_evaluation/results/subset/ragas_results_*.csv`: 38개 subset 결과
- `incorrect_indices.json`: 현재 subset 기준 인덱스

## 출력

- `incorrect_indices.json`: 105개 전체 기준 인덱스로 업데이트
- `incorrect_indices.backup_*.json`: 변환 전 백업 파일

## 변환 로직

1. `user_input` (질문 내용)을 기준으로 subset과 total 매핑
2. subset 인덱스 → total 인덱스로 변환
3. 백업 후 새 파일 저장
