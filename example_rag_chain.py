"""
Simple example demonstrating the RAG with summarization and QA.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorStoreManager

# Load environment
load_dotenv()

print("=" * 80)
print("RAG Example with Summarization and QA")
print("=" * 80)

# Initialize LLM with temperature=0 for deterministic output
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

print("\n1. Initializing vector store...")
vector_store = VectorStoreManager()

print("2. Setting up retriever (similarity search, k=4)...")
retriever = vector_store.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

print("3. Creating QA LLM...")
qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

print("\n" + "=" * 80)
print("Summarization Example")
print("=" * 80)

# Get sample documents
sample_docs = vector_store.vectorstore.get(limit=3)
docs = sample_docs['documents']

if docs:
    print(f"\nSummarizing {len(docs)} document chunks...\n")
    
    summaries = []
    
    for i, chunk in enumerate(docs, 1):
        print(f"Processing chunk {i}/{len(docs)}...")
        summary = llm.invoke(
            f"Summarize the following document in one sentence: {chunk}"
        )
        summaries.append(summary.content.strip())
    
    combined_summary = " ".join(summaries)
    
    print("\n" + "-" * 80)
    print("Individual Summaries:")
    print("-" * 80)
    for i, summary in enumerate(summaries, 1):
        print(f"{i}. {summary}")
    
    print("\n" + "-" * 80)
    print("Combined Summary:")
    print("-" * 80)
    print(combined_summary)

print("\n" + "=" * 80)
print("Q&A Example (Manual RAG)")
print("=" * 80)

# Example query
question = "What is vCenter Server?"
print(f"\nQuestion: {question}\n")

# Step 1: Retrieve relevant documents
print("Retrieving relevant documents...")
relevant_docs = retriever.invoke(question)

print(f"Found {len(relevant_docs)} relevant documents\n")

# Step 2: Build context from retrieved documents
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Step 3: Create prompt with context
prompt = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

# Step 4: Get answer from LLM
print("Generating answer...")
answer = qa_llm.invoke(prompt)

print("Answer:")
print(answer.content)

print("\n" + "-" * 80)
print("Source Documents:")
print("-" * 80)
for i, doc in enumerate(relevant_docs, 1):
    print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"   Content: {doc.page_content[:200]}...")

print("\n" + "=" * 80)
print("✓ Example completed successfully!")
print("=" * 80)
