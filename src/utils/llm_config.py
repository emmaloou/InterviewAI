from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    @staticmethod
    def get_llm(temperature=0.7, model=None):
        """Initialise le LLM"""
        model_name = model or os.getenv("LLM_MODEL", "llama3.1:8b")
        return Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=model_name,
            temperature=temperature
        )
    
    @staticmethod
    def get_embeddings():
        """Initialise les embeddings"""
        return OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        )