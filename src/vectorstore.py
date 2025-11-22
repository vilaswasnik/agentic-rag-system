"""
Vector store management for the Agentic RAG system.
"""
import os
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager:
    """Manage vector store operations using ChromaDB with OpenAI embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "./vectorstore",
        collection_name: str = "agentic_rag_collection",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the OpenAI embedding model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
        """
        # Convert to LangChain Document objects
        langchain_docs = [
            Document(page_content=doc['text'], metadata=doc['metadata'])
            for doc in documents
        ]
        
        # Add to vector store
        self.vectorstore.add_documents(langchain_docs)
        
        print(f"Added {len(documents)} documents to the vector store.")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with metadata
        """
        # Use similarity search
        search_kwargs = {"k": n_results}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata
        
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=n_results
        )
        
        # Format results
        documents = []
        for doc, score in results:
            documents.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return documents
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete and recreate the vector store
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Collection cleared.")
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        # Get the underlying collection to check count
        return len(self.vectorstore.get()['ids'])
