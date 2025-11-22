# System Architecture - OpenAI Integration

## Overview
The Agentic RAG system now uses **ChatOpenAI**, **OpenAIEmbeddings**, and **LangChain Chroma** for enhanced performance and reliability.

## Key Components

### 1. Language Model (LLM)
- **Model**: ChatOpenAI with `gpt-4o-mini`
- **Provider**: OpenAI via `langchain-openai`
- **Usage**: Powers all agent conversations and analysis
- **Benefits**: 
  - Fast response times
  - Cost-effective
  - High quality reasoning

### 2. Embeddings
- **Model**: OpenAIEmbeddings with `text-embedding-3-small`
- **Dimensions**: 1536
- **Provider**: OpenAI
- **Usage**: Converting documents and queries to vector representations
- **Benefits**:
  - Superior semantic understanding
  - Better retrieval accuracy
  - Consistent with OpenAI ecosystem

### 3. Vector Store
- **Framework**: LangChain Chroma
- **Backend**: ChromaDB
- **Storage**: Persistent local storage in `./vectorstore`
- **Features**:
  - Similarity search with scoring
  - Metadata filtering
  - Efficient indexing

## Architecture Flow

```
Document Input (PDF/DOCX/TXT)
    ↓
LangChain Text Splitter (chunks: 1000, overlap: 200)
    ↓
OpenAI Embeddings (text-embedding-3-small)
    ↓
LangChain Chroma Vector Store
    ↓
User Query
    ↓
Query Embedding (text-embedding-3-small)
    ↓
Similarity Search (top-k results with scores)
    ↓
Retrieved Documents
    ↓
AutoGen Agents (powered by ChatOpenAI)
    ├── Orchestrator Agent
    ├── Retriever Agent
    └── Analyzer Agent
    ↓
Final Answer
```

## Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### YAML Config (config.yaml)
```yaml
llm:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 2000

embedding:
  model: "text-embedding-3-small"
  chunk_size: 1000
  chunk_overlap: 200

vectorstore:
  type: "chromadb"
  persist_directory: "./vectorstore"
  collection_name: "agentic_rag_collection"
```

## Agent System

### Orchestrator Agent
- **Role**: Coordinates the workflow
- **LLM**: ChatOpenAI (gpt-4o-mini)
- **Function**: Routes queries and manages agent interactions

### Retriever Agent
- **Role**: Searches vector database
- **LLM**: ChatOpenAI (gpt-4o-mini)
- **Function**: Executes similarity search and retrieves relevant documents

### Analyzer Agent
- **Role**: Analyzes content and generates answers
- **LLM**: ChatOpenAI (gpt-4o-mini)
- **Function**: Synthesizes information from retrieved documents

## Performance Characteristics

### Embeddings
- **Speed**: ~1000 tokens per second
- **Quality**: State-of-the-art semantic understanding
- **Cost**: $0.00002 per 1K tokens (text-embedding-3-small)

### Language Model
- **Speed**: Fast response times with gpt-4o-mini
- **Quality**: High-quality reasoning and generation
- **Cost**: $0.150 per 1M input tokens, $0.600 per 1M output tokens

### Vector Search
- **Speed**: Millisecond-level retrieval
- **Accuracy**: Cosine similarity with normalized scores
- **Scalability**: Handles thousands of documents efficiently

## Testing

Run the test script to verify the system:
```bash
python test_system.py
```

This will:
1. Load the vector store
2. Perform a semantic search
3. Display top results with scores
4. Verify OpenAI integration

## Advantages Over Previous Setup

### Before (Sentence Transformers)
- ❌ Local embeddings (less accurate for complex queries)
- ❌ Separate embedding model to manage
- ❌ Lower semantic understanding quality

### After (OpenAI Embeddings)
- ✅ Cloud-based embeddings (consistent quality)
- ✅ Integrated with OpenAI ecosystem
- ✅ Superior semantic understanding
- ✅ Better retrieval accuracy
- ✅ Simplified deployment (no local model downloads)

## Cost Optimization

### Embedding Costs
- 154 document chunks × ~500 tokens avg = ~77K tokens
- Cost: $0.00002 × 77 = **$0.00154** (one-time indexing)

### Query Costs (per query)
- Query embedding: ~20 tokens = **$0.0000004**
- LLM usage: ~2000 tokens total = **$0.0003**
- **Total per query: ~$0.0003**

### Monthly Estimate (1000 queries)
- Embeddings: $0.0004
- LLM: $0.30
- **Total: ~$0.30/month**

## Best Practices

1. **API Key Security**: Never commit `.env` to version control
2. **Rate Limiting**: OpenAI has rate limits; implement exponential backoff
3. **Caching**: Vector store is persistent; no need to re-embed
4. **Monitoring**: Track token usage in OpenAI dashboard
5. **Error Handling**: Implement retries for API failures

## Troubleshooting

### Issue: "api_key client option must be set"
**Solution**: Ensure `.env` file exists with `OPENAI_API_KEY` set

### Issue: Slow embedding generation
**Solution**: Check internet connection; embeddings require API calls

### Issue: High costs
**Solution**: Use `gpt-4o-mini` instead of `gpt-4`, optimize chunk sizes

## Monitoring & Logging

Track your usage:
- OpenAI Dashboard: https://platform.openai.com/usage
- Local logs: Check terminal output for token counts
- Vector store size: `ls -lh vectorstore/`

---

**System Status**: ✅ Fully operational with OpenAI integration
**Last Updated**: November 21, 2025
