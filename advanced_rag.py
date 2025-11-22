"""
Advanced RAG implementation with document summarization and QA chain.
"""
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.vectorstore import VectorStoreManager
from src.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()


class AdvancedRAGSystem:
    """Advanced RAG system with summarization and QA capabilities."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0,
        search_k: int = 4
    ):
        """
        Initialize Advanced RAG System.
        
        Args:
            model_name: OpenAI model name
            temperature: Temperature for LLM (0 = deterministic)
            search_k: Number of documents to retrieve
        """
        self.model_name = model_name
        self.temperature = temperature
        self.search_k = search_k
        
        # Initialize LLM for summarization
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # Initialize QA LLM
        self.qa_llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize vector store
        self.vector_store = VectorStoreManager()
        
        # Setup retriever
        self.retriever = self.vector_store.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k}
        )
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
    
    def summarize_documents(self, chunks: List[str]) -> str:
        """
        Summarize document chunks into a single combined summary.
        
        Args:
            chunks: List of document text chunks
            
        Returns:
            Combined summary of all chunks
        """
        summaries = []
        
        print(f"Summarizing {len(chunks)} document chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  Processing chunk {i}/{len(chunks)}...", end="\r")
            
            # Generate summary for each chunk
            summary = self.llm.invoke(
                f"Summarize the following document in one sentence: {chunk}"
            )
            summaries.append(summary.content.strip())
        
        print(f"\n✓ Generated {len(summaries)} summaries")
        
        # Combine all summaries
        combined_summary = " ".join(summaries)
        
        return combined_summary
    
    def _create_prompt_template(self) -> str:
        """
        Create a custom prompt template.
        
        Returns:
            Prompt template string
        """
        return """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Get answer from LLM
        answer = self.qa_llm.invoke(prompt)
        
        return {
            "answer": answer.content,
            "source_documents": relevant_docs
        }
    
    def query_with_sources(self, question: str) -> str:
        """
        Query and format answer with sources.
        
        Args:
            question: User question
            
        Returns:
            Formatted answer with sources
        """
        result = self.query(question)
        
        # Format answer
        answer = f"Answer: {result['answer']}\n\n"
        answer += "Sources:\n"
        
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:150] + "..."
            answer += f"{i}. {source}\n   Preview: {content_preview}\n\n"
        
        return answer
    
    def summarize_collection(self) -> str:
        """
        Summarize all documents in the vector store.
        
        Returns:
            Combined summary of all documents
        """
        # Get all documents from vector store
        all_docs = self.vector_store.vectorstore.get()
        
        if not all_docs['documents']:
            return "No documents found in the vector store."
        
        # Summarize all chunks
        combined_summary = self.summarize_documents(all_docs['documents'])
        
        # Generate final meta-summary
        print("\nGenerating final meta-summary...")
        meta_summary = self.llm.invoke(
            f"Summarize the following collection of summaries into a comprehensive overview: {combined_summary}"
        )
        
        return meta_summary.content.strip()


def main():
    """Main function to demonstrate the advanced RAG system."""
    print("=" * 80)
    print("Advanced RAG System with Summarization")
    print("=" * 80)
    
    # Initialize system
    print("\nInitializing Advanced RAG System...")
    rag = AdvancedRAGSystem(
        model_name="gpt-3.5-turbo",
        temperature=0,
        search_k=4
    )
    print("✓ System initialized")
    
    # Example 1: Query with sources
    print("\n" + "=" * 80)
    print("Example 1: Query with Sources")
    print("=" * 80)
    
    question = "What is vCenter Server and how do you install it?"
    print(f"\nQuestion: {question}\n")
    
    answer = rag.query_with_sources(question)
    print(answer)
    
    # Example 2: Summarize specific documents
    print("\n" + "=" * 80)
    print("Example 2: Summarize Sample Documents")
    print("=" * 80)
    
    # Get a few sample chunks
    sample_docs = rag.vector_store.vectorstore.get(limit=5)
    if sample_docs['documents']:
        sample_summary = rag.summarize_documents(sample_docs['documents'])
        print(f"\nSummary of sample documents:\n{sample_summary[:500]}...\n")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Q&A Mode")
    print("Type 'summary' to get a collection summary")
    print("Type 'exit' to quit")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'summary':
                print("\nGenerating collection summary (this may take a moment)...")
                summary = rag.summarize_collection()
                print(f"\nCollection Summary:\n{summary}\n")
                continue
            
            if not user_input:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing...\n")
            answer = rag.query_with_sources(user_input)
            print(answer)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
