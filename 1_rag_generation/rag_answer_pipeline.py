from __future__ import annotations

import gc
import json
import time
import pandas as pd
import torch

from datasets import Dataset
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CURRENT_CONFIG
from utils import load_embeddings
from chunking_strategies import get_chunking_strategy


class RAGPipeline:
    def __init__(self):
        self.config = CURRENT_CONFIG
        self.embeddings = None
        self.chunking_strategy = None
        self.retriever = None
        self.reranker = None
        self._llm_model = None
        self._llm_tokenizer = None

    def load_embeddings(self):
        self.embeddings = load_embeddings()
        return self.embeddings
    
    def load_retriever(self):
        """청킹 전략에 따라 Retriever 로드"""
        if self.embeddings is None:
            self.load_embeddings()
        
        if self.chunking_strategy is None:
            self.chunking_strategy = get_chunking_strategy(self.config)
        
        self.retriever = self.chunking_strategy.get_retriever(self.embeddings)
        return self.retriever

    def load_reranker(self):
        if CrossEncoder is None:
            self.reranker = None
            return None
        model_path = self.config.reranker_model_path
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(model_name, device=device)
        return self.reranker

    def load_llm(self, device: str = "cuda"):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_path, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # bfloat16이 float16보다 안정적 (NaN/inf 문제 감소)
        model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_path,
            device_map="auto" if device == "cuda" else None,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        self._llm_model = model
        self._llm_tokenizer = tokenizer
        return model

    def generate_answer(self, prompt: str) -> str:
        """model.generate()를 직접 호출해 안정적으로 답변 생성"""
        inputs = self._llm_tokenizer(prompt, return_tensors="pt").to(self._llm_model.device)
        with torch.no_grad():
            outputs = self._llm_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self._llm_tokenizer.pad_token_id,
                eos_token_id=self._llm_tokenizer.eos_token_id,
            )
        generated = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 제외하고 답변만 추출
        if "답변:" in generated:
            return generated.split("답변:")[-1].strip()
        return generated[len(prompt):].strip()

    def _rerank(self, question: str, contexts: list[str]) -> list[str]:
        if not self.reranker or not contexts:
            return contexts
        pairs = [(question, c) for c in contexts]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[: self.config.rerank_top_k]]

    def retrieve_contexts(self, question: str) -> list[str]:
        """청킹 전략에 따라 문맥 검색"""
        if self.retriever is None:
            self.load_retriever()
        
        cfg = self.config
        
        # Retriever 타입에 따라 호출 방법 다름
        if hasattr(self.retriever, 'get_relevant_documents'):
            # ParentDocumentRetriever
            docs = self.retriever.get_relevant_documents(question)
            docs = docs[:cfg.retriever_top_k]
        else:
            # 일반 Retriever
            docs = self.retriever.invoke(question)
        
        contexts = [d.page_content for d in docs]
        return self._rerank(question, contexts)

    def build_rag_answer(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = (
            "다음 문맥을 참고하여 질문에 간결하고 정확하게 답하세요.\n\n"
            f"문맥:\n{context_text}\n\n"
            f"질문: {question}\n"
            "답변:"
        )
        return self.generate_answer(prompt)

    def build_dataset(self, df: pd.DataFrame) -> Dataset:
        if self.reranker is None:
            self.load_reranker()
        if self._llm_model is None and self.config.generate_answers:
            self.load_llm()

        # 필터 인덱스 로드 (filter_by_incorrect=True일 때만)
        filter_indices = self._load_filter_indices()
        
        # 처리할 인덱스 목록 계산
        if filter_indices:
            target_indices = sorted(filter_indices)
            print(f"\n{'='*60}")
            print(f" [필터] {len(target_indices)}개 문항만 처리")
            print(f"   인덱스: {target_indices}")
            print(f"{'='*60}\n")
        else:
            target_indices = list(range(len(df)))
            print(f"\n전체 {len(target_indices)}개 문항 처리\n")
        
        total_count = len(target_indices)

        rows = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.config.rag_answers_path
        log_f = log_path.open("w", encoding="utf-8")

        # 진행 상황 추적 변수
        start_time = time.perf_counter()
        processed_count = 0

        for idx in target_indices:
            row = df.iloc[idx]
            question = str(row["Q"]).strip()
            ground_truth = str(row["A"]).strip()
            
            processed_count += 1
            item_start = time.perf_counter()
            
            # 진행 상황 출력
            print(f"[{processed_count}/{total_count}] idx={idx} 처리 중...")
            print(f"  질문: {question[:50]}..." if len(question) > 50 else f"   질문: {question}")
            
            contexts = self.retrieve_contexts(question)

            if self.config.generate_answers and self._llm_model is not None:
                answer = self.build_rag_answer(question, contexts)
            else:
                answer = ground_truth

            # RAGAS 최신 포맷 (2024년 하반기~)
            record = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            }

            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
            
            # 소요시간 및 예상 완료시간 계산
            item_elapsed = time.perf_counter() - item_start
            total_elapsed = time.perf_counter() - start_time
            avg_per_item = total_elapsed / processed_count
            remaining = total_count - processed_count
            eta_seconds = avg_per_item * remaining
            
            # ETA 포맷팅
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}초"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}분"
            else:
                eta_str = f"{eta_seconds/3600:.1f}시간"
            
            print(f"   완료 ({item_elapsed:.1f}초) | 평균: {avg_per_item:.1f}초/건 | ETA: {eta_str} ({remaining}건 남음)")
            print()

        log_f.close()
        
        # 최종 요약
        total_time = time.perf_counter() - start_time
        print(f"{'='*60}")
        print(f" 생성 완료! 총 {processed_count}개, 소요시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"{'='*60}\n")
        
        return Dataset.from_list(rows)

    def _load_filter_indices(self):
        """틀린 문항 인덱스 로드 (filter_by_incorrect=True일 때만)"""
        if not self.config.filter_by_incorrect:
            return None
        
        filter_path = self.config.incorrect_indices_path
        if filter_path.exists():
            with filter_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("incorrect_indices", []))
        else:
            print(f"⚠️  필터 파일이 없습니다: {filter_path}")
            print("   전체 데이터셋으로 진행합니다.")
        return None
    
    def release_llm(self):
        if self._llm_model is None:
            return
        del self._llm_model
        del self._llm_tokenizer
        self._llm_model = None
        self._llm_tokenizer = None
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    cfg = CURRENT_CONFIG
    print(f"{'='*60}")
    print(f"[실험] {cfg.experiment_name}")
    print(f"{'='*60}")
    print(f"Collection: {cfg.collection_name}")
    
    # 청킹 전략 정보 출력
    strategy = get_chunking_strategy(cfg)
    strategy_info = strategy.get_strategy_info()
    print(f"[청킹 전략] {strategy_info['type']}")
    print(f"Retrieval: top_k={cfg.retriever_top_k}, rerank={cfg.rerank_top_k}")
    if getattr(cfg, "use_hybrid_retriever", False):
        print(
            f"Hybrid: vector_top_k={cfg.vector_top_k}, "
            f"bm25_top_k={cfg.bm25_top_k}, rrf_k={cfg.rrf_k}"
        )
    print(f"출력 경로: {cfg.output_dir}\n")
    
    if not cfg.xlsx_path.exists():
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {cfg.xlsx_path}")

    pipeline = RAGPipeline()
    df = pd.read_excel(cfg.xlsx_path)
    dataset = pipeline.build_dataset(df)
    pipeline.release_llm()

    print(f"\n[OK] 답변 생성 완료: {len(dataset)}개")
    print(f"JSONL 저장: {cfg.rag_answers_path}")


if __name__ == "__main__":
    main()
