"""
BM25 인덱스 구축/로드 유틸리티.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi


_kiwi_instance = None
def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        _kiwi_instance = Kiwi()
    return _kiwi_instance


def tokenize(text: str) -> list[str]:
    """한국어 토크나이저 (Kiwi 전용)"""
    text = text.strip().lower()
    if not text:
        return []

    kiwi = _get_kiwi()
    return [t.form for t in kiwi.tokenize(text)]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    items: list[dict]
    tokenized_corpus: list[list[str]]


def build_bm25_index(items: Iterable[dict]) -> BM25Index:
    """items: {child_id, parent_id, text} 리스트"""
    item_list = list(items)
    tokenized_corpus = [tokenize(item["text"]) for item in item_list]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25=bm25, items=item_list, tokenized_corpus=tokenized_corpus)


def save_bm25_index(index: BM25Index, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": index.items,
        "tokenized_corpus": index.tokenized_corpus,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_bm25_index(path: Path | str) -> BM25Index:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    items = payload["items"]
    tokenized_corpus = payload["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25=bm25, items=items, tokenized_corpus=tokenized_corpus)


def bm25_search(index: BM25Index, query: str, top_k: int) -> list[dict]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    # 상위 top_k 인덱스 추출
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked_idx[:top_k]:
        item = index.items[i]
        results.append(
            {
                "child_id": item["child_id"],
                "parent_id": item["parent_id"],
                "text": item["text"],
                "score": float(scores[i]),
            }
        )
    return results


