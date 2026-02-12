from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import pandas as pd

from config import CURRENT_CONFIG
from utils import load_eval_llm, load_env

# 환경 변수 로드
load_env()


def judge_answer_correctness(llm, question: str, response: str, reference: str) -> dict:
    """
    LLM을 사용하여 답변이 정답과 일치하는지 판단
    
    Returns:
        dict: {"is_correct": bool, "explanation": str}
    """
    prompt = f"""다음 질문에 대한 두 답변이 의미적으로 같은 내용인지 판단해주세요.

질문: {question}

생성된 답변: {response}

정답: {reference}

두 답변이 본질적으로 같은 의미를 담고 있으면 "정답", 다르면 "오답"으로 판단하세요.
표현이 조금 다르거나 어미가 달라도 핵심 내용이 같으면 정답입니다.

아래 JSON 형식으로만 답변하세요:
{{
    "judgment": "정답" 또는 "오답",
    "explanation": "판단 근거를 한 문장으로"
}}"""

    try:
        result = llm.invoke(prompt)
        content = result.content.strip()
        
        # JSON 파싱
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        is_correct = data.get("judgment", "오답") == "정답"
        explanation = data.get("explanation", "판단 근거 없음")
        
        return {
            "is_correct": is_correct,
            "explanation": explanation
        }
    except Exception as e:
        print(f"  ⚠️ 판단 오류: {e}")
        return {
            "is_correct": False,
            "explanation": f"판단 실패: {str(e)}"
        }


def calculate_ragas_avg(df: pd.DataFrame) -> dict:
    """RAGAS 메트릭 평균 계산"""
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avg_scores = {}
    
    for metric in metrics:
        if metric in df.columns:
            avg_scores[metric] = df[metric].mean()
    
    return avg_scores


def save_results_to_log(log_dir: Path, avg_scores: dict, acc: float, 
                        correct_count: int, total_count: int, 
                        details: list, csv_filename: str):
    """결과를 마크다운 로그 파일 및 틀린 인덱스 JSON으로 저장"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"evaluation_log_{timestamp}.md"
    
    # 틀린 문항 인덱스 저장 (0-based)
    incorrect_indices = [d["idx"] for d in details if not d["is_correct"]]
    incorrect_path = log_dir / "incorrect_indices.json"
    with incorrect_path.open("w", encoding="utf-8") as f:
        json.dump({
            "incorrect_indices": incorrect_indices,
            "incorrect_count": len(incorrect_indices),
            "total_count": total_count,
            "accuracy": acc
        }, f, indent=2)
    
    print(f"\n❌ 틀린 문항 인덱스 저장: {incorrect_path}")
    print(f"   틀린 문항 수: {len(incorrect_indices)}개")
    
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# 평가 결과 보고서\n\n")
        f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**평가 데이터**: `{csv_filename}`\n\n")
        f.write(f"**총 샘플 수**: {total_count}개\n\n")
        
        f.write(f"---\n\n")
        f.write(f"## 1. RAGAS 메트릭 평균 점수\n\n")
        f.write(f"| 메트릭 | 평균 점수 |\n")
        f.write(f"|--------|----------|\n")
        for metric, score in avg_scores.items():
            f.write(f"| {metric} | {score:.4f} |\n")
        
        f.write(f"\n---\n\n")
        f.write(f"## 2. LLM 기반 정확도 평가\n\n")
        f.write(f"- **정답 개수**: {correct_count}/{total_count}\n")
        f.write(f"- **정확도 (Accuracy)**: {acc:.2%}\n\n")
        
        f.write(f"---\n\n")
        f.write(f"## 3. 샘플별 상세 결과\n\n")
        
        for detail in details:
            idx = detail["idx"]
            question = detail["question"]
            response = detail["response"]
            reference = detail["reference"]
            is_correct = detail["is_correct"]
            explanation = detail["explanation"]
            
            status = "✅ 정답" if is_correct else "❌ 오답"
            
            f.write(f"### 샘플 {idx + 1}: {status}\n\n")
            f.write(f"**질문**: {question}\n\n")
            f.write(f"**생성된 답변**: {response}\n\n")
            f.write(f"**정답**: {reference}\n\n")
            f.write(f"**판단 근거**: {explanation}\n\n")
            f.write(f"---\n\n")
    
    print(f"\n📄 상세 로그 저장: {log_path}")
    return log_path


def main():
    cfg = CURRENT_CONFIG
    
    # CSV 파일 찾기
    csv_files = list(cfg.log_dir.glob("ragas_results_*.csv"))
    if not csv_files:
        print(f"❌ CSV 파일을 찾을 수 없습니다: {cfg.log_dir}")
        return
    
    # 가장 최근 파일 선택
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    print(f"{'='*60}")
    print(f"🔍 LLM 기반 정확도 평가 시작")
    print(f"{'='*60}")
    print(f"실험: {cfg.experiment_name}")
    print(f"평가 모델: {cfg.eval_model}")
    print(f"CSV 파일: {csv_path.name}\n")
    
    # 1. CSV 로드
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"📊 총 {total_samples}개 샘플 로드\n")
    
    # 2. RAGAS 평균 점수 계산
    print(f"--- [RAGAS 평균 점수] ---")
    avg_scores = calculate_ragas_avg(df)
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")
    print()
    
    # 3. LLM 기반 정확도 평가
    print(f"{'='*60}")
    print(f"🤖 LLM 기반 정확도 평가 중...")
    print(f"{'='*60}\n")
    
    llm = load_eval_llm()
    correct_count = 0
    details = []
    
    for idx, row in df.iterrows():
        question = row["user_input"]
        response = row["response"]
        reference = row["reference"]
        
        print(f"[{idx + 1}/{total_samples}] 평가 중...")
        print(f"  질문: {question[:50]}...")
        
        result = judge_answer_correctness(llm, question, response, reference)
        is_correct = result["is_correct"]
        explanation = result["explanation"]
        
        if is_correct:
            correct_count += 1
            print(f"  ✅ 정답")
        else:
            print(f"  ❌ 오답")
        print(f"  📝 {explanation}\n")
        
        details.append({
            "idx": idx,
            "question": question,
            "response": response,
            "reference": reference,
            "is_correct": is_correct,
            "explanation": explanation
        })
    
    # 4. 최종 결과 출력
    accuracy = correct_count / total_samples
    
    print(f"\n{'='*60}")
    print(f"✅ 평가 완료!")
    print(f"{'='*60}")
    print(f"정답: {correct_count}/{total_samples}")
    print(f"정확도 (Accuracy): {accuracy:.2%}\n")
    
    # 5. 로그 저장
    log_path = save_results_to_log(
        log_dir=cfg.log_dir,
        avg_scores=avg_scores,
        acc=accuracy,
        correct_count=correct_count,
        total_count=total_samples,
        details=details,
        csv_filename=csv_path.name
    )
    
    print(f"\n🎉 모든 평가 완료!")


if __name__ == "__main__":
    main()
