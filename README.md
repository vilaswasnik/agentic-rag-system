# Agentic RAG System with AutoGen

An advanced Retrieval-Augmented Generation (RAG) system built with Microsoft's AutoGen framework, featuring autonomous AI agents that collaborate to answer questions based on your documents.

## 🚀 Features

- **Multi-Agent Architecture**: Orchestrator, Retriever, and Analyzer agents working together
- **AutoGen Framework**: Leverages Microsoft AutoGen for agent collaboration
- **OpenAI Integration**: Uses ChatOpenAI (GPT-4o-mini) and OpenAI Embeddings (text-embedding-3-small)
- **Vector Search**: Semantic search using LangChain Chroma with OpenAI embeddings
- **Document Processing**: Supports PDF, DOCX, and TXT file formats
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Interactive CLI**: User-friendly command-line interface for querying

## 📋 System Architecture

The system consists of three main agents:

1. **Orchestrator Agent**: Coordinates the overall workflow and manages agent interactions
2. **Document Retriever Agent**: Searches the vector database for relevant documents
3. **Content Analyzer Agent**: Analyzes retrieved documents and generates comprehensive answers

### Workflow

```
User Query → Orchestrator → Retriever → Vector Search → Retrieved Docs
                    ↓                                          ↓
              Final Answer ← Analyzer ← Document Analysis ←────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd /workspaces/sample.ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📁 Project Structure

```
sample.ai/
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # Document processing and chunking
│   ├── vectorstore.py            # Vector store management
│   └── agents.py                 # AutoGen agent definitions
├── data/
│   └── documents/                # Place your documents here
│       ├── ai_introduction.txt
│       └── rag_explained.txt
├── config.yaml                   # Configuration file
├── main.py                       # Main application script
├── ingest_documents.py           # Document ingestion utility
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Usage

### Step 1: Add Your Documents

Place your documents in the `data/documents/` directory. Supported formats:
- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)

### Step 2: Ingest Documents

Process and index your documents:

```bash
python ingest_documents.py
```

To clear existing documents and start fresh:

```bash
python ingest_documents.py --clear
```

### Step 3: Run the Agentic RAG System

Start the interactive query interface:

```bash
python main.py
```

### Example Interaction

```
================================================================================
Agentic RAG System with AutoGen
================================================================================
Initializing vector store...
Vector store already contains 47 documents.

Initializing Agentic RAG system...
System initialized successfully!

================================================================================
You can now ask questions about your documents.
Type 'exit' or 'quit' to end the session.
================================================================================

Your question: What is RAG and how does it work?

Processing your query...

[Agent interactions occur here...]

Answer: RAG (Retrieval-Augmented Generation) is a technique that combines 
large language models with external knowledge retrieval. It works by...
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

### LLM Settings
```yaml
llm:
  model: "gpt-4o-mini"  # or "gpt-4o", "gpt-4", etc.
  temperature: 0.7
  max_tokens: 2000
```

### Embedding Configuration
```yaml
embedding:
  model: "text-embedding-3-small"  # or "text-embedding-3-large"
  chunk_size: 1000
  chunk_overlap: 200
```

### Vector Store Settings
```yaml
vectorstore:
  type: "chromadb"
  persist_directory: "./vectorstore"
  collection_name: "agentic_rag_collection"
```

## 🎯 Key Components

### Document Processor
- Handles multiple file formats (PDF, DOCX, TXT)
- Implements intelligent text chunking with LangChain
- Preserves document metadata

### Vector Store Manager
- LangChain Chroma integration for vector storage
- OpenAI embeddings (text-embedding-3-small)
- Semantic similarity search with score ranking

### AutoGen Agents
- **RetrieverAgent**: Searches and retrieves relevant documents
- **AnalyzerAgent**: Analyzes content using ChatOpenAI
- **OrchestratorAgent**: Coordinates multi-agent workflow
- Function calling for document retrieval

## 🔧 Advanced Usage

### Custom Document Processing

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
documents = processor.process_directory("path/to/documents")
```

### Direct Vector Store Access

```python
from src.vectorstore import VectorStoreManager

vectorstore = VectorStoreManager()
results = vectorstore.search("your query", n_results=10)
```

### Programmatic Querying

```python
from src.agents import AgenticRAGSystem

rag_system = AgenticRAGSystem(vectorstore, llm_config)
answer = rag_system.query("What is machine learning?")
```

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution**: Make sure you've created a `.env` file with your API key.

### Issue: No documents found
**Solution**: Run `python ingest_documents.py` to process documents first.

### Issue: Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📚 Dependencies

- **pyautogen**: Multi-agent framework
- **langchain & langchain-openai**: LLM and embeddings integration
- **langchain-chroma**: Vector database with LangChain
- **chromadb**: Vector storage backend
- **openai**: OpenAI API client
- **PyPDF2**: PDF processing
- **python-docx**: Word document processing

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional document format support
- More sophisticated chunking strategies
- Agent memory and conversation history
- Web UI interface
- Multiple LLM provider support

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Microsoft AutoGen framework
- ChromaDB for vector storage
- OpenAI for LLM capabilities
- The open-source AI community

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Built with ❤️ using AutoGen and modern AI technologies**