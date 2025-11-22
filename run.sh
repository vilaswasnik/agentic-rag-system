#!/bin/bash
# Quick start script for the Agentic RAG system

echo "=============================================="
echo "  Agentic RAG System - Quick Start"
echo "=============================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env file with your OPENAI_API_KEY"
    exit 1
fi

# Check if vector store exists
if [ ! -d vectorstore ]; then
    echo "⚠️  Vector store not found. Running document ingestion..."
    python ingest_documents.py
    echo ""
fi

echo "Choose an option:"
echo ""
echo "1. Test System (quick verification)"
echo "2. Simple RAG Example (with summarization)"
echo "3. Advanced RAG System (interactive mode)"
echo "4. Main System (AutoGen multi-agent)"
echo "5. Re-index Documents"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running system test..."
        python test_system.py
        ;;
    2)
        echo ""
        echo "Running simple RAG example..."
        python example_rag_chain.py
        ;;
    3)
        echo ""
        echo "Starting advanced RAG system..."
        echo "(Type 'exit' to quit, 'summary' for collection summary)"
        python advanced_rag.py
        ;;
    4)
        echo ""
        echo "Starting main system with AutoGen agents..."
        python main.py
        ;;
    5)
        echo ""
        echo "Re-indexing documents..."
        python ingest_documents.py --clear
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-5."
        exit 1
        ;;
esac
