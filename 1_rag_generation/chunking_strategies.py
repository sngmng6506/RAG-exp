"""
청킹 전략 클래스들 (Strategy Pattern)

Parent-Child Chunking:
  - ParentDocStore: Parent 문서를 JSON으로 디스크에 저장/조회
  - ParentChildRetriever: Child로 검색 → Parent 반환
  - ParentChildChunkingStrategy: 위 컴포넌트를 조합한 전략
"""
from __future__ import annotations

import json
import uuid

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from langchain_chroma import Chroma
from langchain.vectorstores import Chroma  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.documents import Document
from langchain.schema import Document  # type: ignore
from bm25_index import bm25_search, load_bm25_index

# =============================================================================
# Parent-Child 컴포넌트
# =============================================================================

class ParentDocStore:
    """
    Parent 문서를 JSON 파일로 디스크에 저장/조회하는 클래스.
    
    - 프로세스 간 영속화 지원 (build → retrieve 분리 실행 가능)
    - key: parent_id (UUID)
    - value: Document 직렬화 (page_content + metadata)
    """
    
    def __init__(self, store_path: Path | str):
        self.store_path = Path(store_path)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()
    
    def _load(self):
        """디스크에서 데이터 로드"""
        if self.store_path.exists():
            with self.store_path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
    
    def _save(self):
        """디스크에 데이터 저장"""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
    
    def add(self, parent_id: str, document: Document):
        """Parent 문서 추가"""
        self._data[parent_id] = {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
    
    def get(self, parent_id: str) -> Document | None:
        """Parent 문서 조회"""
        if parent_id not in self._data:
            return None
        item = self._data[parent_id]
        return Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
    
    def save(self):
        """변경사항 디스크에 저장"""
        self._save()
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __contains__(self, parent_id: str) -> bool:
        return parent_id in self._data

    def items(self):
        """(parent_id, Document) 반복자"""
        for parent_id, item in self._data.items():
            yield parent_id, Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )


class ParentChildRetriever:
    """
    Child 청크로 검색하고 Parent 청크를 반환하는 Retriever.
    
    동작 방식:
    1. Vectorstore(Child)에서 유사도 검색
    2. 검색된 Child의 metadata['parent_id']로 Parent 조회
    3. 중복 제거 후 Parent 문서 반환
    
    langchain Retriever 인터페이스와 호환되도록 invoke(), get_relevant_documents() 제공.
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        parent_store: ParentDocStore,
        search_k: int = 10,
    ):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.search_k = search_k
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        """Child 검색 → Parent 반환 (중복 제거)"""
        # 1. Child 검색 (더 많이 검색해서 다양한 Parent 확보)
        child_docs = self.vectorstore.similarity_search(query, k=self.search_k * 2)
        
        # 2. Parent ID 추출 (순서 유지, 중복 제거)
        seen_parent_ids: set[str] = set()
        unique_parent_ids: list[str] = []
        
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                unique_parent_ids.append(parent_id)
        
        # 3. Parent 문서 조회
        parent_docs: list[Document] = []
        for parent_id in unique_parent_ids[:self.search_k]:
            parent = self.parent_store.get(parent_id)
            if parent:
                parent_docs.append(parent)
        
        return parent_docs
    
    def invoke(self, query: str) -> list[Document]:
        """langchain Retriever 인터페이스 호환"""
        return self.get_relevant_documents(query)


class HybridParentChildRetriever:
    """
    Vector + BM25 하이브리드 검색(RRF) 후 Parent 문서를 반환하는 Retriever.
    """
    def __init__(
        self,
        vectorstore: Chroma,
        parent_store: ParentDocStore,
        bm25_index_path: Path,
        vector_top_k: int,
        bm25_top_k: int,
        rrf_k: int,
        final_top_k: int,
    ):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.bm25_index_path = bm25_index_path
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self._bm25_index = None

    def _load_bm25(self):
        if self._bm25_index is None:
            if not self.bm25_index_path.exists():
                raise FileNotFoundError(
                    f"BM25 인덱스가 없습니다: {self.bm25_index_path}\n"
                    f"build_pdf_chroma.py를 실행하여 BM25 인덱스를 생성하세요."
                )
            self._bm25_index = load_bm25_index(self.bm25_index_path)

    def _rrf_merge(self, vector_docs: list[Document], bm25_results: list[dict]) -> list[dict]:
        """
        child_id 기준으로 RRF 점수를 계산.
        반환: [{"child_id", "parent_id", "score"}] 정렬 리스트
        """
        scores: dict[str, float] = {}
        parent_map: dict[str, str] = {}

        # Vector 결과 (순위 기반)
        for rank, doc in enumerate(vector_docs, start=1):
            child_id = doc.metadata.get("child_id")
            parent_id = doc.metadata.get("parent_id")
            if not child_id or not parent_id:
                continue
            scores[child_id] = scores.get(child_id, 0.0) + 1.0 / (self.rrf_k + rank)
            parent_map[child_id] = parent_id

        # BM25 결과 (순위 기반)
        for rank, item in enumerate(bm25_results, start=1):
            child_id = item.get("child_id")
            parent_id = item.get("parent_id")
            if not child_id or not parent_id:
                continue
            scores[child_id] = scores.get(child_id, 0.0) + 1.0 / (self.rrf_k + rank)
            parent_map[child_id] = parent_id

        merged = [
            {"child_id": child_id, "parent_id": parent_map[child_id], "score": score}
            for child_id, score in scores.items()
            if child_id in parent_map
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Vector + BM25 하이브리드 검색 후 Parent 반환"""
        self._load_bm25()

        vector_docs = self.vectorstore.similarity_search(query, k=self.vector_top_k)
        bm25_results = bm25_search(self._bm25_index, query, top_k=self.bm25_top_k)
        merged = self._rrf_merge(vector_docs, bm25_results)

        # Parent 문서 반환 (중복 제거)
        seen_parent_ids: set[str] = set()
        parent_docs: list[Document] = []
        for item in merged:
            parent_id = item["parent_id"]
            if parent_id in seen_parent_ids:
                continue
            parent = self.parent_store.get(parent_id)
            if parent:
                parent_docs.append(parent)
                seen_parent_ids.add(parent_id)
            if len(parent_docs) >= self.final_top_k:
                break
        return parent_docs

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


class ParentChildBuilder:
    """
    원본 문서에서 Parent-Child 구조를 생성하는 빌더.
    
    동작 방식:
    1. Parent Splitter로 큰 청크 생성 → UUID 부여 → ParentDocStore에 저장
    2. Child Splitter로 작은 청크 생성 → metadata에 parent_id 추가
    3. Child 청크들을 Vectorstore에 저장
    """
    
    def __init__(
        self,
        parent_splitter: RecursiveCharacterTextSplitter,
        child_splitter: RecursiveCharacterTextSplitter,
    ):
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter
    
    def build(
        self,
        documents: list[Document],
        vectorstore: Chroma,
        parent_store: ParentDocStore,
    ) -> dict[str, Any]:
        """
        Parent-Child 구조 생성 및 저장.
        
        Returns:
            {"parent_count": int, "child_count": int}
        """
        all_child_docs: list[Document] = []
        parent_count = 0
        
        for doc in documents:
            # 1. Parent 청크 생성
            parent_chunks = self.parent_splitter.split_documents([doc])
            
            for parent_chunk in parent_chunks:
                # 2. Parent에 UUID 부여 및 저장
                parent_id = str(uuid.uuid4())
                parent_chunk.metadata["parent_id"] = parent_id
                parent_store.add(parent_id, parent_chunk)
                parent_count += 1
                
                # 3. Child 청크 생성 (Parent 기반)
                child_chunks = self.child_splitter.split_documents([parent_chunk])
                
                # 4. Child에 parent_id/child_id 메타데이터 추가
                for child in child_chunks:
                    child.metadata["child_id"] = str(uuid.uuid4())
                    child.metadata["parent_id"] = parent_id
                    all_child_docs.append(child)
        
        # 5. ParentDocStore 저장
        parent_store.save()
        
        # 6. Child 청크들을 Vectorstore에 추가
        if all_child_docs:
            vectorstore.add_documents(all_child_docs)
        
        return {
            "parent_count": parent_count,
            "child_count": len(all_child_docs),
            "child_docs": all_child_docs,
        }


# =============================================================================
# 청킹 전략 클래스들
# =============================================================================

class BaseChunkingStrategy(ABC):
    """청킹 전략 기본 클래스"""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def build_vectorstore(self, documents, embeddings):
        """Vector DB 구축"""
        pass
    
    @abstractmethod
    def get_retriever(self, embeddings):
        """Retriever 로드 (답변 생성 시)"""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> dict:
        """전략 정보 반환 (로깅용)"""
        pass


class SimpleChunkingStrategy(BaseChunkingStrategy):
    """일반 청킹 전략: 단순 분할 → 검색"""
    
    def build_vectorstore(self, documents, embeddings):
        cfg = self.config
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(cfg.chroma_dir),
            collection_name=cfg.collection_name,
        )
        
        return {
            "vectorstore": vectorstore,
            "chunk_count": len(splits),
            "strategy": "simple",
        }
    
    def get_retriever(self, embeddings):
        cfg = self.config
        
        vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=embeddings,
            collection_name=cfg.collection_name,
        )
        
        return vectorstore.as_retriever(search_kwargs={"k": cfg.retriever_top_k})
    
    def get_strategy_info(self) -> dict:
        cfg = self.config
        return {
            "type": "Simple Chunking",
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
        }


class ParentChildChunkingStrategy(BaseChunkingStrategy):
    """
    Parent-Child 청킹 전략:
    - 작은 Child 청크로 검색 (정밀도 향상)
    - 큰 Parent 청크를 반환 (문맥 보존)
    """
    
    def _get_parent_store_path(self) -> Path:
        """ParentDocStore JSON 파일 경로"""
        return self.config.docstore_path / "parent_docs.json"
    
    def build_vectorstore(self, documents, embeddings):
        cfg = self.config
        
        # 1. Splitter 생성
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.parent_chunk_size,
            chunk_overlap=cfg.parent_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.child_chunk_size,
            chunk_overlap=cfg.child_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        # 2. Vectorstore 생성 (Child 저장용)
        cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embeddings,
            persist_directory=str(cfg.chroma_dir),
        )
        
        # 3. ParentDocStore 생성
        cfg.docstore_path.mkdir(parents=True, exist_ok=True)
        parent_store = ParentDocStore(self._get_parent_store_path())
        
        # 4. Parent-Child 구조 생성
        print("Parent-Child 구조 생성 중...")
        builder = ParentChildBuilder(parent_splitter, child_splitter)
        counts = builder.build(documents, vectorstore, parent_store)
        
        print(f"  - Parent 청크: {counts['parent_count']}개")
        print(f"  - Child 청크: {counts['child_count']}개")
        
        return {
            "vectorstore": vectorstore,
            "chunk_count": counts["child_count"],
            "parent_count": counts["parent_count"],
            "strategy": "parent_child",
            "docstore_path": cfg.docstore_path,
            "child_docs": counts["child_docs"],
        }
    
    def get_retriever(self, embeddings):
        cfg = self.config
        
        # 1. Vectorstore 로드 (Child)
        vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=embeddings,
            collection_name=cfg.collection_name,
        )
        
        # 2. ParentDocStore 로드
        store_path = self._get_parent_store_path()
        if not store_path.exists():
            raise FileNotFoundError(
                f"ParentDocStore가 없습니다: {store_path}\n"
                f"먼저 build_pdf_chroma.py를 실행하여 Parent-Child 구조를 생성하세요."
            )
        parent_store = ParentDocStore(store_path)
        
        # 3. Retriever 생성 (하이브리드 여부에 따라 선택)
        if getattr(cfg, "use_hybrid_retriever", False):
            return HybridParentChildRetriever(
                vectorstore=vectorstore,
                parent_store=parent_store,
                bm25_index_path=cfg.bm25_index_path,
                vector_top_k=cfg.vector_top_k,
                bm25_top_k=cfg.bm25_top_k,
                rrf_k=cfg.rrf_k,
                final_top_k=cfg.retriever_top_k,
            )
        return ParentChildRetriever(
            vectorstore=vectorstore,
            parent_store=parent_store,
            search_k=cfg.retriever_top_k,
        )
    
    def get_strategy_info(self) -> dict:
        cfg = self.config
        return {
            "type": "Parent-Child Chunking",
            "parent_chunk_size": cfg.parent_chunk_size,
            "parent_chunk_overlap": cfg.parent_chunk_overlap,
            "child_chunk_size": cfg.child_chunk_size,
            "child_chunk_overlap": cfg.child_chunk_overlap,
            "docstore_path": str(cfg.docstore_path),
        }


# =============================================================================
# Factory
# =============================================================================

def get_chunking_strategy(config) -> BaseChunkingStrategy:
    """설정에 따라 적절한 청킹 전략 반환"""
    if config.use_parent_child:
        return ParentChildChunkingStrategy(config)
    else:
        return SimpleChunkingStrategy(config)
