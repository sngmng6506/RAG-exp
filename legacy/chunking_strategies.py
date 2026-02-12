"""
청킹 전략 클래스들 (Strategy Pattern)
"""
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore


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
    """일반 청킹 전략"""
    
    def build_vectorstore(self, documents, embeddings):
        """일반 방식: 단순 청킹"""
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
        """일반 Vectorstore Retriever 반환"""
        cfg = self.config
        
        vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=embeddings,
            collection_name=cfg.collection_name,
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": cfg.retriever_top_k}
        )
        
        return retriever
    
    def get_strategy_info(self) -> dict:
        """전략 정보"""
        cfg = self.config
        return {
            "type": "Simple Chunking",
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
        }


class ParentChildChunkingStrategy(BaseChunkingStrategy):
    """Parent-Child 청킹 전략"""
    
    def build_vectorstore(self, documents, embeddings):
        """Parent-Child 방식: 작은 child로 검색, 큰 parent로 반환"""
        cfg = self.config
        
        # Child splitter (검색용, 작은 청크)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.child_chunk_size,
            chunk_overlap=cfg.child_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        # Parent splitter (반환용, 큰 청크)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.parent_chunk_size,
            chunk_overlap=cfg.parent_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        # Vectorstore (child chunks 저장)
        vectorstore = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embeddings,
            persist_directory=str(cfg.chroma_dir),
        )
        
        # Docstore (parent chunks 영구 저장)
        docstore_path = cfg.docstore_path
        docstore_path.mkdir(parents=True, exist_ok=True)
        docstore = LocalFileStore(str(docstore_path))
        
        # ParentDocumentRetriever 생성
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        # 문서 추가 (자동으로 parent-child 구조 생성)
        print("Parent-Child 구조 생성 중...")
        retriever.add_documents(documents, ids=None)
        
        # Child 개수 확인
        child_count = vectorstore._collection.count()
        
        return {
            "vectorstore": vectorstore,
            "retriever": retriever,
            "chunk_count": child_count,
            "strategy": "parent_child",
            "docstore_path": docstore_path,
        }
    
    def get_retriever(self, embeddings):
        """ParentDocumentRetriever 반환"""
        cfg = self.config
        
        # Vectorstore 로드 (child chunks)
        vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=embeddings,
            collection_name=cfg.collection_name,
        )
        
        # Docstore 로드 (parent chunks)
        docstore_path = cfg.docstore_path
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"Docstore가 없습니다: {docstore_path}\n"
                f"먼저 build_pdf_chroma.py를 실행하여 Parent-Child 구조를 생성하세요."
            )
        docstore = LocalFileStore(str(docstore_path))
        
        # Splitter 설정 (build 시와 동일하게)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.child_chunk_size,
            chunk_overlap=cfg.child_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.parent_chunk_size,
            chunk_overlap=cfg.parent_chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        # ParentDocumentRetriever 생성
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        return retriever
    
    def get_strategy_info(self) -> dict:
        """전략 정보"""
        cfg = self.config
        return {
            "type": "Parent-Child Chunking",
            "parent_chunk_size": cfg.parent_chunk_size,
            "parent_chunk_overlap": cfg.parent_chunk_overlap,
            "child_chunk_size": cfg.child_chunk_size,
            "child_chunk_overlap": cfg.child_chunk_overlap,
            "docstore_path": str(cfg.docstore_path),
        }


# Factory 함수
def get_chunking_strategy(config) -> BaseChunkingStrategy:
    """설정에 따라 적절한 청킹 전략 반환"""
    if config.use_parent_child:
        return ParentChildChunkingStrategy(config)
    else:
        return SimpleChunkingStrategy(config)
