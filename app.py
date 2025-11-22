"""
Graphical User Interface for the Agentic RAG System using Gradio.
"""
import gradio as gr
from advanced_rag import AdvancedRAGSystem
import time
import os
import shutil
from pathlib import Path
from src.document_processor import DocumentProcessor

# Initialize the RAG system globally
print("Initializing RAG System...")
rag_system = AdvancedRAGSystem(
    model_name="gpt-3.5-turbo",
    temperature=0,
    search_k=1
)
print("✓ RAG System ready!")

# Initialize document processor
doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# Ensure data directory exists
DATA_DIR = "/workspaces/sample.ai/data/documents"
os.makedirs(DATA_DIR, exist_ok=True)


def format_sources(source_docs):
    """Format source documents for display."""
    if not source_docs:
        return "No sources found."
    
    # Show only the first (most relevant) source
    doc = source_docs[0]
    source = doc.metadata.get('source', 'Unknown')
    filename = source.split('/')[-1]
    preview = doc.page_content[:200].replace('\n', ' ') + "..."
    sources_text = f"**📄 {filename}**\n\n"
    sources_text += f"_{preview}_\n"
    return sources_text


def query_rag(question, history):
    """Process a query and return the answer with word-by-word streaming."""
    if not question or not question.strip():
        history.append({"role": "assistant", "content": "⚠️ Please enter a question."})
        yield history
        return
    
    # Add user question to history
    history.append({"role": "user", "content": question})
    yield history
    
    try:
        # Get answer from RAG system
        result = rag_system.query(question)
        
        # Stream the answer word by word
        answer_header = "**Answer:**\n\n"
        words = result['answer'].split()
        current_answer = answer_header
        
        # Add empty assistant message
        history.append({"role": "assistant", "content": ""})
        
        # Stream words one by one
        for i, word in enumerate(words):
            current_answer += word + " "
            history[-1] = {"role": "assistant", "content": current_answer}
            yield history
            time.sleep(0.05)  # Small delay between words for smooth streaming
        
        # Add sources after answer is complete
        sources_text = "\n\n---\n\n**Sources:**\n\n" + format_sources(result['source_documents'])
        history[-1] = {"role": "assistant", "content": current_answer + sources_text}
        yield history
        
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield history


def summarize_documents(progress=gr.Progress()):
    """Generate a summary of all documents in the collection."""
    try:
        progress(0, desc="Initializing...")
        
        # Get all documents
        all_docs = rag_system.vector_store.vectorstore.get()
        if not all_docs['documents']:
            return "⚠️ No documents found in the vector store."
        
        total = min(len(all_docs['documents']), 10)  # Limit to 10 for demo
        summaries = []
        
        progress(0, desc=f"Summarizing {total} documents...")
        
        for i, chunk in enumerate(all_docs['documents'][:total]):
            progress((i + 1) / total, desc=f"Processing chunk {i+1}/{total}...")
            summary = rag_system.llm.invoke(
                f"Summarize the following document in one sentence: {chunk}"
            )
            summaries.append(summary.content.strip())
        
        combined = " ".join(summaries)
        
        progress(1.0, desc="Generating meta-summary...")
        meta_summary = rag_system.llm.invoke(
            f"Summarize the following collection of summaries into a comprehensive overview: {combined}"
        )
        
        result = f"**Collection Summary** ({total} documents analyzed)\n\n"
        result += meta_summary.content.strip()
        
        return result
        
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"


def upload_and_process_documents(files, progress=gr.Progress()):
    """Upload and process new documents."""
    if not files:
        return "⚠️ No files uploaded."
    
    try:
        progress(0, desc="Starting upload...")
        results = []
        total_files = len(files)
        
        for idx, file in enumerate(files):
            progress((idx / total_files), desc=f"Processing file {idx+1}/{total_files}...")
            
            # Get the file path from the uploaded file
            file_path = file.name
            filename = Path(file_path).name
            
            # Copy to data directory
            destination = os.path.join(DATA_DIR, filename)
            shutil.copy2(file_path, destination)
            
            # Process the document
            try:
                chunks = doc_processor.process_document(destination)
                
                # Add to vector store
                documents = [
                    {
                        'text': chunk,
                        'metadata': {'source': destination, 'chunk_id': i, 'file_name': filename}
                    }
                    for i, chunk in enumerate(chunks)
                ]
                
                rag_system.vector_store.add_documents(documents)
                
                results.append(f"✅ **{filename}**: {len(chunks)} chunks added")
            except Exception as e:
                results.append(f"❌ **{filename}**: {str(e)}")
        
        progress(1.0, desc="Complete!")
        
        # Get updated stats
        doc_count = rag_system.vector_store.get_collection_count()
        summary = f"### 📊 Upload Complete\n\n"
        summary += f"**Total chunks in system:** {doc_count}\n\n"
        summary += "### Results:\n\n"
        summary += "\n".join(results)
        
        return summary
        
    except Exception as e:
        return f"❌ Error during upload: {str(e)}"


def get_stats():
    """Get system statistics."""
    try:
        doc_count = rag_system.vector_store.get_collection_count()
        
        # List uploaded files
        files_in_data = []
        if os.path.exists(DATA_DIR):
            files_in_data = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
        
        files_list = "\n".join([f"  - {f}" for f in files_in_data]) if files_in_data else "  - None"
        
        return f"""
### 📊 System Statistics

- **Documents Indexed:** {doc_count} chunks
- **Files in System:** {len(files_in_data)}
- **Model:** gpt-3.5-turbo
- **Embeddings:** text-embedding-3-small
- **Search Type:** Similarity (top-1)
- **Temperature:** 0 (deterministic)

### 📁 Uploaded Files:
{files_list}
        """
    except Exception as e:
        return f"Error retrieving stats: {str(e)}"


# Create the Gradio interface
with gr.Blocks(title="Agentic RAG System") as demo:
    
    gr.Markdown(
        """
        # 🤖 Agentic RAG System
        ### Intelligent Document Q&A powered by OpenAI and ChromaDB
        
        Ask questions about your VMware VCP-DCV Lab Manual and other documents!
        """
    )
    
    with gr.Tabs():
        
        # Tab 1: Chat Interface
        with gr.Tab("💬 Ask Questions"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is vCenter Server? How do I install ESXi?",
                    scale=4
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
            
            gr.Examples(
                examples=[
                    ["What is vCenter Server?"],
                    ["How do I install ESXi?"],
                    ["Explain VMware vSphere architecture"],
                    ["What are the lab requirements?"],
                    ["How do I configure virtual machines?"],
                ],
                inputs=question_input,
                label="Example Questions"
            )
            
            clear_btn = gr.Button("Clear Conversation")
            
            # Event handlers with streaming (auto-detected via generator)
            submit_btn.click(
                fn=query_rag,
                inputs=[question_input, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                outputs=question_input
            )
            
            question_input.submit(
                fn=query_rag,
                inputs=[question_input, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                outputs=question_input
            )
            
            clear_btn.click(lambda: [], outputs=chatbot)
        
        # Tab 2: Upload Documents
        with gr.Tab("📤 Upload Documents"):
            gr.Markdown(
                """
                ### Upload New Documents
                
                Upload PDF, DOCX, or TXT files to automatically process and add them to the RAG system.
                The documents will be:
                1. Split into chunks
                2. Embedded using OpenAI embeddings
                3. Stored in ChromaDB vector store
                4. Immediately available for queries
                
                **Supported formats:** PDF (.pdf), Word (.docx), Text (.txt)
                """
            )
            
            with gr.Row():
                file_upload = gr.File(
                    label="Select Files to Upload",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"]
                )
            
            upload_btn = gr.Button("Upload & Process", variant="primary", size="lg")
            upload_output = gr.Markdown(label="Upload Status")
            
            upload_btn.click(
                fn=upload_and_process_documents,
                inputs=file_upload,
                outputs=upload_output
            )
        
        # Tab 3: Document Summary
        with gr.Tab("📄 Document Summary"):
            gr.Markdown("Generate a comprehensive summary of all documents in the collection.")
            
            summary_btn = gr.Button("Generate Summary", variant="primary")
            summary_output = gr.Markdown(label="Summary")
            
            summary_btn.click(
                fn=summarize_documents,
                outputs=summary_output
            )
        
        # Tab 4: System Info
        with gr.Tab("ℹ️ System Info"):
            stats_output = gr.Markdown(value=get_stats())
            refresh_btn = gr.Button("🔄 Refresh Stats")
            
            refresh_btn.click(
                fn=get_stats,
                outputs=stats_output
            )
            
            gr.Markdown(
                """
                ### 🎯 How to Use
                
                1. **Ask Questions**: Go to the "Ask Questions" tab and type your question
                2. **View Sources**: Each answer includes source documents with previews
                3. **Generate Summary**: Get an overview of all documents in the collection
                
                ### 📚 Available Documents
                
                - VMware VCP-DCV Lab Manual (Complete)
                - AI & Machine Learning Introduction
                - RAG (Retrieval-Augmented Generation) Guide
                
                ### 💡 Tips
                
                - Be specific in your questions for better answers
                - Use the example questions to get started
                - Sources are provided with each answer for verification
                - The system uses semantic search to find relevant information
                
                ### ⚙️ Technical Details
                
                - **Vector Store**: ChromaDB with persistent storage
                - **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
                - **LLM**: GPT-3.5-turbo with temperature=0 for consistency
                - **Retrieval**: Top-1 most relevant document per query
                - **Chain Type**: "Stuff" - all context in single prompt
                """
            )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True
    )
