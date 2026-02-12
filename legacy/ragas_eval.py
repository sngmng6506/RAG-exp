from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from config import CURRENT_CONFIG
from utils import load_embeddings, load_eval_llm, load_env

# 환경 변수 로드
load_env()


# ===============================
# Loaders
# ===============================
def load_dataset_from_jsonl() -> Dataset:
    """JSONL 파일에서 RAGAS 평가용 Dataset 생성 (구버전/신버전 포맷 모두 지원)"""
    jsonl_path = CURRENT_CONFIG.rag_answers_path
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 파일이 없습니다: {jsonl_path}")

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                
                # 구버전 포맷(question, answer, ground_truth, contexts)을
                # 신버전 포맷(user_input, response, reference, retrieved_contexts)으로 변환
                if "question" in data:
                    data["user_input"] = data.pop("question")
                if "answer" in data and "response" not in data:
                    data["response"] = data["answer"]
                if "ground_truth" in data:
                    data["reference"] = data.pop("ground_truth")
                if "contexts" in data:
                    data["retrieved_contexts"] = data.pop("contexts")
                
                rows.append(data)

    print(f"📂 JSONL 로드 완료: {len(rows)}개 샘플")
    return Dataset.from_list(rows)


# ===============================
# Checkpoint 관리
# ===============================
def load_checkpoint():
    """체크포인트 파일 로드 (이미 평가 완료된 인덱스 목록)"""
    checkpoint_path = CURRENT_CONFIG.checkpoint_path
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_indices": [], "results": [], "start_time": None}


def save_checkpoint(checkpoint_data):
    """체크포인트 저장"""
    log_dir = CURRENT_CONFIG.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CURRENT_CONFIG.checkpoint_path
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def evaluate_single_sample(sample_data, embeddings, eval_llm, metrics, run_config):
    """단일 샘플 평가"""
    dataset = Dataset.from_list([sample_data])
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    return result.to_pandas().iloc[0].to_dict()


# ===============================
# Main
# ===============================
def main():
    cfg = CURRENT_CONFIG
    print(f"{'='*60}")
    print(f"🚀 RAGAS 평가 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"실험: {cfg.experiment_name}")
    print(f"평가 모델: {cfg.eval_model}")
    print(f"Log: {cfg.log_dir}\n")
    
    # 1) JSONL에서 데이터셋 로드
    dataset = load_dataset_from_jsonl()
    total_samples = len(dataset)
    
    # 2) 체크포인트 로드
    checkpoint = load_checkpoint()
    completed_indices = set(checkpoint["completed_indices"])
    results = checkpoint["results"]
    
    if completed_indices:
        print(f"📌 체크포인트 발견: {len(completed_indices)}/{total_samples} 샘플 완료")
        print(f"   ➡️  {len(completed_indices)}번부터 재개합니다.\n")
    else:
        print(f"📊 전체 {total_samples}개 샘플 평가 시작\n")
        checkpoint["start_time"] = datetime.now().isoformat()
    
    # 3) 평가용 모델 로드
    print("🔧 모델 로딩 중...")
    embeddings = load_embeddings()
    eval_llm = load_eval_llm()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(timeout=300, max_retries=5, max_wait=60, max_workers=1)
    print("✅ 모델 로딩 완료\n")
    
    # 4) 샘플별 평가 (체크포인트 기반 재개)
    start_time = time.perf_counter()
    
    for idx in range(total_samples):
        if idx in completed_indices:
            continue  # 이미 완료된 샘플은 건너뛰기
        
        sample = dataset[idx]
        print(f"[{idx+1}/{total_samples}] 평가 중...")
        print(f"  📝 질문: {sample['user_input'][:50]}...")
        
        try:
            sample_start = time.perf_counter()
            result = evaluate_single_sample(sample, embeddings, eval_llm, metrics, run_config)
            sample_elapsed = time.perf_counter() - sample_start
            
            # 결과에 인덱스 추가
            result["sample_idx"] = idx
            results.append(result)
            
            # 체크포인트 업데이트
            checkpoint["completed_indices"].append(idx)
            checkpoint["results"] = results
            save_checkpoint(checkpoint)
            
            # 진행 상황 출력
            print(f"  ✅ 완료 ({sample_elapsed:.1f}초)")
            print(f"     - faithfulness: {result.get('faithfulness', 'N/A'):.3f}")
            print(f"     - answer_relevancy: {result.get('answer_relevancy', 'N/A'):.3f}")
            print(f"     - context_precision: {result.get('context_precision', 'N/A'):.3f}")
            print(f"     - context_recall: {result.get('context_recall', 'N/A'):.3f}")
            print()
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            print(f"     중단된 위치: {idx}번 샘플")
            print(f"     체크포인트 저장됨. 재실행하면 이어서 진행됩니다.\n")
            raise
    
    total_elapsed = time.perf_counter() - start_time
    
    # 5) 최종 결과 저장
    print(f"\n{'='*60}")
    print(f"✅ 전체 평가 완료!")
    print(f"{'='*60}")
    print(f"총 소요시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")
    print(f"샘플당 평균: {total_elapsed/total_samples:.1f}초\n")
    
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = cfg.log_dir / f"ragas_results_{timestamp}.csv"
    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"📁 결과 저장: {out_csv}")
    
    # 평균 점수 출력
    print(f"\n--- [평균 점수] ---")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df_results.columns:
            avg_score = df_results[metric].mean()
            print(f"{metric}: {avg_score:.4f}")
    
    # 체크포인트 파일 삭제 (완료되었으므로)
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"\n🗑️  체크포인트 파일 삭제 완료")


if __name__ == "__main__":
    main()
