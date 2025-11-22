"""
Utility script to ingest documents into the vector store.
"""
import os
import sys
from dotenv import load_dotenv
import yaml
from src.document_processor import DocumentProcessor
from src.vectorstore import VectorStoreManager


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def ingest_documents(documents_dir: str = "data/documents", clear_existing: bool = False):
    """
    Ingest documents into the vector store.
    
    Args:
        documents_dir: Directory containing documents to ingest
        clear_existing: Whether to clear existing documents
    """
    print("=" * 80)
    print("Document Ingestion Tool")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Initialize vector store
    vectorstore_config = config['vectorstore']
    embedding_config = config['embedding']
    
    vectorstore = VectorStoreManager(
        persist_directory=vectorstore_config['persist_directory'],
        collection_name=vectorstore_config['collection_name']
    )
    
    # Clear existing documents if requested
    if clear_existing:
        print("\nClearing existing documents...")
        vectorstore.clear_collection()
    
    # Check current document count
    current_count = vectorstore.get_collection_count()
    print(f"\nCurrent documents in vector store: {current_count}")
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        print(f"\nError: Documents directory '{documents_dir}' not found.")
        print("Please create the directory and add documents to ingest.")
        sys.exit(1)
    
    # Process documents
    print(f"\nProcessing documents from '{documents_dir}'...")
    
    processor = DocumentProcessor(
        chunk_size=embedding_config['chunk_size'],
        chunk_overlap=embedding_config['chunk_overlap']
    )
    
    documents = processor.process_directory(documents_dir)
    
    if not documents:
        print("\nNo documents found to process.")
        print("Supported formats: .pdf, .docx, .txt")
        sys.exit(0)
    
    # Add documents to vector store
    print(f"\nAdding {len(documents)} document chunks to vector store...")
    vectorstore.add_documents(documents)
    
    # Show final count
    final_count = vectorstore.get_collection_count()
    print(f"\nIngestion complete!")
    print(f"Total documents in vector store: {final_count}")
    print(f"Newly added: {final_count - current_count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--documents-dir",
        type=str,
        default="data/documents",
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before ingesting new ones"
    )
    
    args = parser.parse_args()
    
    ingest_documents(args.documents_dir, args.clear)
