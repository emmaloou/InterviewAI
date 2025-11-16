from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import os

class VectorStore:
    def __init__(self, embeddings, persist_directory="./data/vector_db"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def create_or_load_store(self, collection_name: str):
        """Crée ou charge une collection"""
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict] = None):
        """Ajoute des documents au vector store"""
        vectorstore = self.create_or_load_store(collection_name)
        
        # Split texts
        documents = [Document(page_content=text, metadata=meta or {}) 
                    for text, meta in zip(texts, metadatas or [{}]*len(texts))]
        
        splits = self.text_splitter.split_documents(documents)
        
        # Add to vectorstore
        vectorstore.add_documents(splits)
        
        return vectorstore
    
    def similarity_search(self, collection_name: str, query: str, k: int = 3):
        """Recherche de documents similaires"""
        vectorstore = self.create_or_load_store(collection_name)
        return vectorstore.similarity_search(query, k=k)