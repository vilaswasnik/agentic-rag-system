# RAG System Implementation Guide

## Complete Implementation with ChatOpenAI and Summarization

Your RAG system now includes advanced features with proper implementation of:
- **ChatOpenAI** for LLM
- **OpenAIEmbeddings** for vectorization
- **LangChain Chroma** for vector storage
- **Document Summarization** capability
- **Manual RAG chain** implementation

## Key Implementation Components

### 1. LLM Configuration
```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0  # Deterministic output
)
```

### 2. Document Summarization
```python
summaries = []

for chunk in docs:
    summary = llm.invoke(
        f"Summarize the following document in one sentence: {chunk}"
    )
    summaries.append(summary.content.strip())

combined_summary = " ".join(summaries)
```

### 3. Retriever Setup
```python
retriever = vector_store.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 4. QA LLM
```python
qa_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)
```

### 5. Manual RAG Chain ("stuff" approach)
```python
# Retrieve documents
relevant_docs = retriever.invoke(question)

# Build context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Create prompt
prompt = f"""Use the following pieces of context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

# Get answer
answer = qa_llm.invoke(prompt)
```

## Files Created

### 1. `example_rag_chain.py`
Simple example demonstrating:
- Document summarization with `llm.invoke()`
- Retriever with `similarity` search and `k=4`
- Manual RAG implementation (stuff approach)
- Answer generation with sources

**Run it:**
```bash
python example_rag_chain.py
```

### 2. `advanced_rag.py`
Advanced system with:
- `AdvancedRAGSystem` class
- Batch document summarization
- Collection-wide summarization
- Interactive Q&A mode
- Query with sources functionality

**Run it:**
```bash
python advanced_rag.py
```

## Usage Examples

### Basic Summarization
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

summaries = []
for chunk in docs:
    summary = llm.invoke(
        f"Summarize the following document in one sentence: {chunk}"
    )
    summaries.append(summary.content.strip())

combined_summary = " ".join(summaries)
```

### Basic RAG Query
```python
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorStoreManager

# Setup
vector_store = VectorStoreManager()
retriever = vector_store.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Query
question = "What is vCenter Server?"
relevant_docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in relevant_docs])

prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
answer = qa_llm.invoke(prompt)

print(answer.content)
```

### Advanced RAG with Class
```python
from advanced_rag import AdvancedRAGSystem

# Initialize
rag = AdvancedRAGSystem(
    model_name="gpt-3.5-turbo",
    temperature=0,
    search_k=4
)

# Query with sources
answer = rag.query_with_sources("What is vCenter Server?")
print(answer)

# Summarize documents
summary = rag.summarize_collection()
print(summary)
```

## Chain Type: "stuff"

The "stuff" approach means:
1. **Retrieve** relevant documents from vector store
2. **Stuff** all documents into a single prompt
3. **Send** everything to the LLM at once
4. **Generate** answer based on all context

**Pros:**
- Simple and straightforward
- One LLM call
- Good for small number of documents

**Cons:**
- Limited by context window
- All documents must fit in prompt

## Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `gpt-3.5-turbo` | Fast, cost-effective OpenAI model |
| `temperature` | `0` | Deterministic output (no randomness) |
| `search_type` | `similarity` | Use cosine similarity for retrieval |
| `search_kwargs` | `{"k": 4}` | Return top 4 most relevant documents |
| `chain_type` | `stuff` | Put all docs in one prompt |

## Testing

### Test RAG System
```bash
python test_system.py
```

### Test Simple Example
```bash
python example_rag_chain.py
```

### Test Advanced System
```bash
python advanced_rag.py
```

## Example Output

### Summarization
```
Processing chunk 1/3...
Processing chunk 2/3...
Processing chunk 3/3...

Combined Summary:
The document provides instructions for accessing and setting up 
a lab environment for installing and configuring VMware vSphere...
```

### Q&A with Sources
```
Question: What is vCenter Server?

Answer:
vCenter Server is a centralized management tool for VMware 
virtualized environments.

Sources:
1. data/documents/vcneter_update_VCP_DCV_Lab_Manual_Updated.pdf
   Preview: 5. Put vCenter Name and IP and click on Add...
```

## Cost Estimates

### Per Query (with summarization)
- Summarization (3 chunks): $0.0001
- Retrieval embedding: $0.000001
- QA generation: $0.0003
- **Total: ~$0.0004 per query**

### Per Query (without summarization)
- Retrieval embedding: $0.000001
- QA generation: $0.0003
- **Total: ~$0.0003 per query**

## Best Practices

1. **Use temperature=0** for factual Q&A
2. **Set appropriate k value** (4 is good default)
3. **Monitor token usage** to control costs
4. **Cache summaries** to avoid re-summarizing
5. **Use gpt-3.5-turbo** for cost efficiency

## Integration with Your Code

### Import and Use
```python
from advanced_rag import AdvancedRAGSystem

# Initialize once
rag = AdvancedRAGSystem(
    model_name="gpt-3.5-turbo",
    temperature=0,
    search_k=4
)

# Use multiple times
answer1 = rag.query("Question 1")
answer2 = rag.query("Question 2")
```

### Customize
```python
# Change model
rag = AdvancedRAGSystem(model_name="gpt-4", temperature=0.3)

# Change retrieval count
rag = AdvancedRAGSystem(search_k=10)

# Access components
rag.llm  # Summarization LLM
rag.qa_llm  # QA LLM
rag.retriever  # Document retriever
rag.vector_store  # Vector store manager
```

## Architecture

```
Document Chunks
    ↓
OpenAI Embeddings (text-embedding-3-small)
    ↓
LangChain Chroma Vector Store
    ↓
User Query → Embedding → Similarity Search → Top K Documents
    ↓
Context Building (join all documents)
    ↓
Prompt Template (stuff approach)
    ↓
ChatOpenAI (gpt-3.5-turbo, temp=0)
    ↓
Answer + Sources
```

## Next Steps

1. ✅ System fully implemented with your specifications
2. ✅ Summarization working
3. ✅ RAG chain operational
4. ✅ Examples provided

You can now:
- Run `python example_rag_chain.py` for simple example
- Run `python advanced_rag.py` for interactive mode
- Import `AdvancedRAGSystem` in your own code
- Customize parameters as needed

---

**Status**: ✅ Fully implemented and tested
**Last Updated**: November 21, 2025
