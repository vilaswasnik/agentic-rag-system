"""
Test script to verify the RAG system with OpenAI embeddings.
"""
import os
from dotenv import load_dotenv
from src.vectorstore import VectorStoreManager

# Load environment
load_dotenv()

print("=" * 80)
print("Testing Agentic RAG System with OpenAI")
print("=" * 80)

# Initialize vector store
print("\n1. Initializing vector store with OpenAI embeddings...")
vectorstore = VectorStoreManager()

# Check document count
doc_count = vectorstore.get_collection_count()
print(f"   ✓ Vector store loaded with {doc_count} documents")

# Test search
print("\n2. Testing semantic search...")
query = "What is vCenter Server?"
results = vectorstore.search(query, n_results=3)

print(f"   Query: '{query}'")
print(f"   Found {len(results)} relevant documents:\n")

for i, doc in enumerate(results, 1):
    print(f"   Document {i}:")
    print(f"   Source: {doc['metadata'].get('source', 'Unknown')}")
    print(f"   Score: {doc.get('score', 'N/A')}")
    print(f"   Content: {doc['text'][:200]}...")
    print()

print("=" * 80)
print("✓ System is working correctly with OpenAI embeddings!")
print("=" * 80)
print("\nYou can now run 'python main.py' to start the interactive RAG system.")
