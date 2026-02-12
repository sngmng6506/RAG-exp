import time
import uuid


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from config import CURRENT_CONFIG
from bm25_index import build_bm25_index, save_bm25_index
from chunking_strategies import ParentDocStore


def main():
    cfg = CURRENT_CONFIG
    if not cfg.use_parent_child:
        raise ValueError("BM25 인덱스는 Parent-Child 모드에서만 지원합니다.")

    store_path = cfg.docstore_path / "parent_docs.json"
    if not store_path.exists():
        raise FileNotFoundError(
            f"ParentDocStore가 없습니다: {store_path}\n"
            f"먼저 build_pdf_chroma.py를 실행하세요."
        )

    print(f"{'='*60}")
    print(f"[BM25 인덱스 생성] {cfg.experiment_name}")
    print(f"{'='*60}")
    print(f"Docstore: {store_path}")
    print(f"BM25 저장 경로: {cfg.bm25_index_path}")
    print(
        f"Child Splitter: size={cfg.child_chunk_size}, overlap={cfg.child_chunk_overlap}"
    )

    # 이미 BM25 인덱스가 존재하면 스킵
    if cfg.skip_bm25_if_exists and cfg.bm25_index_path.exists():
        print("BM25 인덱스가 이미 존재합니다. 생성을 건너뜁니다.")
        print(f" - 경로: {cfg.bm25_index_path}")
        return

    parent_store = ParentDocStore(store_path)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    start = time.perf_counter()
    bm25_items = []
    parent_count = 0
    child_count = 0

    for parent_id, parent_doc in parent_store.items():
        parent_count += 1
        child_docs = child_splitter.split_documents([parent_doc])
        for child in child_docs:
            child_id = str(uuid.uuid4())
            bm25_items.append(
                {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "text": child.page_content,
                }
            )
            child_count += 1

    if not bm25_items:
        raise ValueError("BM25 인덱스 생성용 Child 청크가 없습니다.")

    index = build_bm25_index(bm25_items)
    save_bm25_index(index, cfg.bm25_index_path)
    elapsed = time.perf_counter() - start

    print(f"\nParent 문서 수: {parent_count}")
    print(f"Child 청크 수: {child_count}")
    print(f"BM25 인덱스 저장 완료: {cfg.bm25_index_path}")
    print(f"소요시간: {elapsed:.2f}초")


if __name__ == "__main__":
    main()
