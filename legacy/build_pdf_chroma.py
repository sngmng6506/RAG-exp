import time
from langchain_community.document_loaders import PyPDFLoader

from config import CURRENT_CONFIG
from utils import load_embeddings, get_dir_size_bytes, format_bytes
from chunking_strategies import get_chunking_strategy


def load_pdf_documents():
    pdf_dir = CURRENT_CONFIG.pdf_dir
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일이 없습니다: {pdf_dir}")

    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)
    return documents




def main():
    cfg = CURRENT_CONFIG
    print(f"{'='*60}")
    print(f"🔧 실험: {cfg.experiment_name}")
    print(f"{'='*60}")
    print(f"PDF 폴더: {cfg.pdf_dir}")
    print(f"저장 위치: {cfg.chroma_dir}")
    print(f"Collection: {cfg.collection_name}")
    print(f"임베딩 모델: {cfg.embedding_model_path}")
    
    # 청킹 전략 로드
    strategy = get_chunking_strategy(cfg)
    strategy_info = strategy.get_strategy_info()
    
    print(f"\n📋 청킹 전략: {strategy_info['type']}")
    for key, value in strategy_info.items():
        if key != 'type':
            print(f"  {key}: {value}")

    model_size_bytes = get_dir_size_bytes(cfg.embedding_model_path)
    print(f"\n임베딩 모델 용량: {format_bytes(model_size_bytes)}")

    # 문서 로드
    embeddings = load_embeddings()
    documents = load_pdf_documents()

    # 청킹 전략에 따라 Vector DB 구축
    start = time.perf_counter()
    result = strategy.build_vectorstore(documents, embeddings)
    elapsed = time.perf_counter() - start
    
    # 결과 출력
    print(f"\n원본 문서 수: {len(documents)}")
    print(f"청크 수: {result['chunk_count']}")
    print(f"컬렉션 이름: {result['vectorstore']._collection.name}")
    
    if result['strategy'] == 'parent_child':
        print(f"Docstore 경로: {result['docstore_path']}")
    
    print(f"임베딩 소요시간: {elapsed:.2f}초")
    print("✅ 임베딩 및 저장 완료")


if __name__ == "__main__":
    main()