"""
공통 유틸리티 함수
"""
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
except Exception:
    torch = None

import httpx
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from config import CURRENT_CONFIG

DISABLE_SSL_VERIFY = True


def load_embeddings():
    """임베딩 모델 로드"""
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model_path = CURRENT_CONFIG.embedding_model_path
    model_name = str(model_path) if model_path.exists() else "ko-sbert-sts"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_eval_llm():
    """평가용 OpenAI LLM 로드"""
    http_client = httpx.Client(verify=False) if DISABLE_SSL_VERIFY else None
    return ChatOpenAI(
        model=CURRENT_CONFIG.eval_model,
        temperature=0,
        max_retries=5,
        request_timeout=60,
        http_client=http_client,
    )


def get_dir_size_bytes(path: Path) -> int:
    """디렉토리 크기 계산 (바이트)"""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def format_bytes(num_bytes: int) -> str:
    """바이트를 읽기 쉬운 형식으로 변환"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def load_env():
    """환경 변수 로드 및 검증"""
    from dotenv import load_dotenv
    
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    
    if not os.getenv("OPENAI_API_KEY"):
        print(f"⚠️ OPENAI_API_KEY가 없습니다. .env 파일 확인: {env_path}")
    
    return env_path
