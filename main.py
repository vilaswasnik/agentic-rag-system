"""
Main script for the Agentic RAG system.
"""
import os
import sys
from dotenv import load_dotenv
import yaml
from src.document_processor import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.agents import AgenticRAGSystem


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_environment():
    """Setup environment variables."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return api_key


def initialize_vectorstore(config: dict, documents_dir: str = "data/documents"):
    """Initialize vector store and ingest documents."""
    print("Initializing vector store...")
    
    vectorstore_config = config['vectorstore']
    embedding_config = config['embedding']
    
    vectorstore = VectorStoreManager(
        persist_directory=vectorstore_config['persist_directory'],
        collection_name=vectorstore_config['collection_name']
    )
    
    # Check if documents need to be processed
    doc_count = vectorstore.get_collection_count()
    if doc_count == 0:
        print(f"No documents found in vector store. Processing documents from {documents_dir}...")
        
        if not os.path.exists(documents_dir):
            print(f"Warning: Documents directory '{documents_dir}' not found.")
            print("Please add documents to the 'data/documents' directory.")
            return vectorstore
        
        processor = DocumentProcessor(
            chunk_size=embedding_config['chunk_size'],
            chunk_overlap=embedding_config['chunk_overlap']
        )
        
        documents = processor.process_directory(documents_dir)
        if documents:
            vectorstore.add_documents(documents)
            print(f"Successfully processed and indexed {len(documents)} document chunks.")
        else:
            print("No documents found to process.")
    else:
        print(f"Vector store already contains {doc_count} documents.")
    
    return vectorstore


def main():
    """Main function to run the Agentic RAG system."""
    print("=" * 80)
    print("Agentic RAG System with AutoGen")
    print("=" * 80)
    
    # Setup
    try:
        api_key = setup_environment()
        config = load_config()
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)
    
    # Initialize vector store
    vectorstore = initialize_vectorstore(config)
    
    # Setup LLM configuration
    llm_config = {
        "model": config['llm']['model'],
        "api_key": api_key,
        "temperature": config['llm']['temperature'],
    }
    
    # Initialize Agentic RAG system
    print("\nInitializing Agentic RAG system...")
    rag_system = AgenticRAGSystem(
        vectorstore=vectorstore,
        llm_config=llm_config,
        max_round=10
    )
    print("System initialized successfully!")
    
    # Interactive query loop
    print("\n" + "=" * 80)
    print("You can now ask questions about your documents.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("\nThank you for using the Agentic RAG system. Goodbye!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing your query...\n")
            print("-" * 80)
            
            # Process query
            answer = rag_system.query(question)
            
            print("-" * 80)
            print(f"\nAnswer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
