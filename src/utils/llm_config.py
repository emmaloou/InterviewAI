import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class LLMConfig:
    @staticmethod
    def _ensure_openai_key():
        """Vérifie que la clé OpenAI est disponible"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY est requis pour initialiser le modèle. "
                "Ajoutez-le dans votre fichier .env."
            )
        return api_key

    @staticmethod
    def get_llm(temperature=0.3, model=None, callbacks=None, max_tokens=512):
        """Initialise le LLM OpenAI"""
        api_key = LLMConfig._ensure_openai_key()
        model_name = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    @staticmethod
    def get_embeddings(model: str | None = None):
        """Initialise les embeddings OpenAI"""
        api_key = LLMConfig._ensure_openai_key()
        embedding_model = model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

        return OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model,
            base_url=os.getenv("OPENAI_API_BASE"),
        )