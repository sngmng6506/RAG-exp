"""
incorrect_indices.json 수정 스크립트

subset(38개) 기준의 인덱스를 전체(105개) 기준의 인덱스로 변환합니다.

사용법:
    python fix_incorrect_indices.py

입력:
    - 3_re_evaluation/results/total/ragas_results_*.csv (105개 전체 결과)
    - 3_re_evaluation/results/subset/ragas_results_*.csv (38개 subset 결과)
    - incorrect_indices.json (현재 subset 기준 인덱스)

출력:
    - incorrect_indices.json (105개 전체 기준 인덱스로 업데이트)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def find_latest_csv(directory: Path, pattern: str = "ragas_results_*.csv") -> Path:
    """디렉토리에서 가장 최근 CSV 파일을 찾습니다."""
    csv_files = list(directory.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def main():
    # 경로 설정
    exp_root = Path(__file__).resolve().parent.parent
    
    print("=" * 60)
    print("incorrect_indices.json 수정 스크립트")
    print("=" * 60)
    
    # 1. 전체 결과 (105개) 로드
    total_dir = exp_root / "3_re_evaluation" / "results" / "total"
    total_csv = find_latest_csv(total_dir)
    df_total = pd.read_csv(total_csv)
    print(f"\n[1] 전체 결과 로드: {total_csv.name}")
    print(f"    샘플 수: {len(df_total)}개")
    
    # 2. Subset 결과 (38개) 로드
    subset_dir = exp_root / "3_re_evaluation" / "results" / "subset"
    subset_csv = find_latest_csv(subset_dir)
    df_subset = pd.read_csv(subset_csv)
    print(f"\n[2] Subset 결과 로드: {subset_csv.name}")
    print(f"    샘플 수: {len(df_subset)}개")
    
    # 3. 현재 incorrect_indices.json 로드
    incorrect_json = exp_root / "incorrect_indices.json"
    if not incorrect_json.exists():
        print(f"\n[ERROR] {incorrect_json} 파일이 없습니다.")
        return
    
    with incorrect_json.open("r", encoding="utf-8") as f:
        old_data = json.load(f)
    
    old_incorrect = old_data.get("incorrect_indices", [])
    old_total_count = old_data.get("total_count", len(old_incorrect))
    
    print(f"\n[3] 현재 incorrect_indices.json 로드")
    print(f"    오답 인덱스 수: {len(old_incorrect)}개")
    print(f"    기준 total_count: {old_total_count}")
    print(f"    인덱스: {old_incorrect}")
    
    # 이미 전체 기준인지 확인
    if old_total_count == len(df_total):
        print(f"\n[INFO] 이미 전체 기준({len(df_total)}개)으로 설정되어 있습니다.")
        confirm = input("그래도 다시 변환하시겠습니까? (y/n): ").strip().lower()
        if confirm != 'y':
            print("취소되었습니다.")
            return
    
    # 4. user_input -> total 인덱스 매핑 생성
    print(f"\n[4] user_input 기준으로 매핑 생성 중...")
    user_input_to_total_idx = {}
    for i, row in df_total.iterrows():
        user_input = row.get("user_input", "")
        sample_idx = row.get("sample_idx", i)
        user_input_to_total_idx[user_input] = sample_idx
    
    # 5. subset 인덱스 -> total 인덱스 매핑
    subset_to_total = {}
    for i, row in df_subset.iterrows():
        user_input = row.get("user_input", "")
        subset_idx = row.get("sample_idx", i)
        if user_input in user_input_to_total_idx:
            subset_to_total[subset_idx] = user_input_to_total_idx[user_input]
    
    print(f"    매핑된 항목 수: {len(subset_to_total)}개")
    
    # 6. subset 오답 인덱스 -> total 오답 인덱스 변환
    print(f"\n[5] 인덱스 변환 중...")
    new_incorrect = []
    unmapped = []
    
    for subset_idx in old_incorrect:
        if subset_idx in subset_to_total:
            new_incorrect.append(subset_to_total[subset_idx])
        else:
            unmapped.append(subset_idx)
            print(f"    [WARN] subset index {subset_idx} 매핑 없음")
    
    new_incorrect = sorted(new_incorrect)
    
    print(f"\n[6] 변환 결과")
    print(f"    변환된 인덱스 수: {len(new_incorrect)}개")
    print(f"    매핑 실패: {len(unmapped)}개")
    print(f"    새 인덱스: {new_incorrect}")
    
    # 7. 백업 생성
    backup_path = incorrect_json.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with backup_path.open("w", encoding="utf-8") as f:
        json.dump(old_data, f, indent=2, ensure_ascii=False)
    print(f"\n[7] 백업 저장: {backup_path.name}")
    
    # 8. 새 incorrect_indices.json 저장
    total_count = len(df_total)
    new_data = {
        "incorrect_indices": new_incorrect,
        "incorrect_count": len(new_incorrect),
        "total_count": total_count,
        "accuracy": (total_count - len(new_incorrect)) / total_count
    }
    
    with incorrect_json.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[8] incorrect_indices.json 업데이트 완료")
    print(f"    total_count: {new_data['total_count']}")
    print(f"    incorrect_count: {new_data['incorrect_count']}")
    print(f"    accuracy: {new_data['accuracy']:.2%}")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
