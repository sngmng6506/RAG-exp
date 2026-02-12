"""
두 RAGAS 결과를 병합하여 재계산하는 스크립트

1. 전체 RAGAS 결과 (total/ragas_results_20260205_111120.csv) - 105개 baseline
2. 재평가 결과 (subset/ragas_results_20260206_095319.csv) - 38개 Parent-Child

로직:
- user_input을 기준으로 매칭 (sample_idx가 아닌 질문 내용으로 매칭)
- 재평가 결과에 있는 문항 → 재평가 결과로 대체
- 재평가 결과에 없는 문항 → 전체 결과 그대로 유지
- 최종 평균 재계산
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def main():
    # 경로 설정
    exp_root = Path(__file__).resolve().parent.parent
    
    # 1. 전체 RAGAS 결과 로드 (105개 - 원래 baseline 전체 평가)
    total_csv = exp_root / "3_re_evaluation" / "results" / "total" / "ragas_results_20260205_111120.csv"
    df_total = pd.read_csv(total_csv)
    print(f"[INFO] 전체 RAGAS 결과 (baseline): {len(df_total)}개 샘플")
    
    # 2. 재평가 결과 로드 (38개 - Parent-Child로 재평가)
    subset_csv = exp_root / "3_re_evaluation" / "results" / "subset" / "ragas_results_20260206_095319.csv"
    df_subset = pd.read_csv(subset_csv)
    print(f"[INFO] 재평가 결과 (Parent-Child): {len(df_subset)}개 샘플")
    
    # 메트릭 컬럼
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    # 3. user_input을 기준으로 재평가 결과를 딕셔너리로 변환
    # user_input -> 메트릭 값
    subset_metrics = {}
    for _, row in df_subset.iterrows():
        user_input = row.get("user_input", "")
        if user_input:
            subset_metrics[user_input] = {
                col: row[col] for col in metric_cols if col in row and pd.notna(row[col])
            }
    
    print(f"[INFO] user_input 기준 재평가 결과: {len(subset_metrics)}개")
    
    # 4. 전체 결과를 순회하면서 병합
    merged_results = []
    replaced_count = 0
    
    for i, row in df_total.iterrows():
        user_input = row.get("user_input", "")
        
        result = {
            "sample_idx": row.get("sample_idx", i),
            "user_input": user_input,
            "source": "original"  # 출처 표시
        }
        
        # 해당 문항이 재평가 결과에 있으면 대체
        if user_input in subset_metrics:
            for col in metric_cols:
                # 재평가 결과가 있으면 사용, 없으면 원본 사용
                result[col] = subset_metrics[user_input].get(col, row.get(col, 0))
            result["source"] = "re-evaluated"
            replaced_count += 1
        else:
            # 재평가 결과에 없으면 원본 그대로
            for col in metric_cols:
                result[col] = row.get(col, 0)
        
        merged_results.append(result)
    
    print(f"[INFO] 대체된 문항 수: {replaced_count}개")
    
    # DataFrame으로 변환
    df_merged = pd.DataFrame(merged_results)
    
    # 5. 평균 계산
    print(f"\n{'='*60}")
    print("[RESULT] 병합 결과 요약")
    print(f"{'='*60}")
    
    print(f"\n총 샘플 수: {len(df_merged)}")
    print(f"원본 유지 (baseline): {len(df_merged[df_merged['source'] == 'original'])}개")
    print(f"재평가 대체 (Parent-Child): {len(df_merged[df_merged['source'] == 're-evaluated'])}개")
    
    print(f"\n--- [원본 baseline 전체 평균] ---")
    for col in metric_cols:
        if col in df_total.columns:
            print(f"{col}: {df_total[col].mean():.4f}")
    
    print(f"\n--- [재평가 subset 평균 (참고)] ---")
    for col in metric_cols:
        if col in df_subset.columns:
            print(f"{col}: {df_subset[col].mean():.4f}")
    
    print(f"\n--- [병합 후 RAGAS 평균] ---")
    for col in metric_cols:
        if col in df_merged.columns:
            print(f"{col}: {df_merged[col].mean():.4f}")
    
    # 변화량 출력
    print(f"\n--- [변화량 (baseline -> 병합)] ---")
    for col in metric_cols:
        if col in df_total.columns and col in df_merged.columns:
            original = df_total[col].mean()
            merged = df_merged[col].mean()
            diff = merged - original
            direction = "+" if diff > 0 else "" if diff < 0 else " "
            print(f"{col}: {original:.4f} -> {merged:.4f} ({direction}{diff:.4f})")
    
    # 6. 결과 저장
    output_dir = exp_root / "3_re_evaluation" / "results" / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = output_dir / f"merged_ragas_results_{timestamp}.csv"
    df_merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 병합 결과 저장: {output_csv}")
    
    # 요약 JSON 저장
    summary = {
        "total_samples": len(df_merged),
        "original_kept": len(df_merged[df_merged['source'] == 'original']),
        "re_evaluated": len(df_merged[df_merged['source'] == 're-evaluated']),
        "baseline_avg": {col: float(df_total[col].mean()) for col in metric_cols if col in df_total.columns},
        "subset_avg": {col: float(df_subset[col].mean()) for col in metric_cols if col in df_subset.columns},
        "merged_avg": {col: float(df_merged[col].mean()) for col in metric_cols if col in df_merged.columns},
        "timestamp": timestamp
    }
    
    summary_json = output_dir / f"merge_summary_{timestamp}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] 요약 저장: {summary_json}")


if __name__ == "__main__":
    main()
