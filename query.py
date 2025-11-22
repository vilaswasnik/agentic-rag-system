"""
Simple interactive script to query your RAG system.
"""
from advanced_rag import AdvancedRAGSystem

print("=" * 70)
print("  Agentic RAG System - Interactive Mode")
print("=" * 70)
print("\nInitializing system...")

# Initialize the RAG system
rag = AdvancedRAGSystem(
    model_name="gpt-3.5-turbo",
    temperature=0,
    search_k=4
)

print("✓ System ready!")
print("\nCommands:")
print("  - Type your question to get an answer")
print("  - Type 'summary' to summarize all documents")
print("  - Type 'exit' or 'quit' to stop")
print("=" * 70)

while True:
    try:
        # Get user input
        question = input("\n💬 Your question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Check for empty input
        if not question:
            print("⚠️  Please enter a question.")
            continue
        
        # Handle summary command
        if question.lower() == 'summary':
            print("\n🔄 Generating collection summary (this may take a minute)...")
            summary = rag.summarize_collection()
            print("\n" + "=" * 70)
            print("📄 COLLECTION SUMMARY:")
            print("=" * 70)
            print(summary)
            print("=" * 70)
            continue
        
        # Process regular question
        print("\n🔍 Searching and generating answer...")
        result = rag.query(question)
        
        # Display answer
        print("\n" + "=" * 70)
        print("💡 ANSWER:")
        print("=" * 70)
        print(result['answer'])
        
        # Display sources
        print("\n" + "-" * 70)
        print("📚 SOURCES:")
        print("-" * 70)
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', 'Unknown')
            # Show just filename
            filename = source.split('/')[-1]
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"{i}. {filename}")
            print(f"   Preview: {preview}...")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try again.")
