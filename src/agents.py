"""
AutoGen agents for the Agentic RAG system.
"""
import os
from typing import List, Dict, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorStoreManager


class RetrieverAgent:
    """Agent responsible for retrieving relevant documents."""
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        llm_config: Dict,
        name: str = "DocumentRetriever"
    ):
        """
        Initialize retriever agent.
        
        Args:
            vectorstore: Vector store manager instance
            llm_config: LLM configuration
            name: Agent name
        """
        self.vectorstore = vectorstore
        self.name = name
        
        system_message = """You are a document retriever agent. Your job is to:
1. Understand user queries
2. Search the vector database for relevant documents
3. Return the most relevant document chunks

When you receive a query, use the search_documents function to retrieve relevant information.
Always provide the source information with your retrieved documents."""
        
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
        )
    
    def search_documents(self, query: str, n_results: int = 5) -> str:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Formatted string with retrieved documents
        """
        results = self.vectorstore.search(query, n_results)
        
        if not results:
            return "No relevant documents found."
        
        output = f"Found {len(results)} relevant documents:\n\n"
        for i, doc in enumerate(results, 1):
            output += f"Document {i}:\n"
            output += f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            output += f"Content: {doc['text']}\n"
            output += "-" * 80 + "\n\n"
        
        return output


class AnalyzerAgent:
    """Agent responsible for analyzing retrieved documents."""
    
    def __init__(
        self,
        llm_config: Dict,
        name: str = "ContentAnalyzer"
    ):
        """
        Initialize analyzer agent.
        
        Args:
            llm_config: LLM configuration
            name: Agent name
        """
        self.name = name
        
        system_message = """You are a content analyzer agent. Your job is to:
1. Analyze retrieved documents
2. Extract relevant information
3. Synthesize a comprehensive answer to the user's question

You should:
- Focus on accuracy and relevance
- Cite sources when appropriate
- Provide clear and concise answers
- Acknowledge when information is insufficient"""
        
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
        )


class OrchestratorAgent:
    """Agent responsible for orchestrating the RAG workflow."""
    
    def __init__(
        self,
        llm_config: Dict,
        name: str = "Orchestrator"
    ):
        """
        Initialize orchestrator agent.
        
        Args:
            llm_config: LLM configuration
            name: Agent name
        """
        self.name = name
        
        system_message = """You are an orchestrator agent. Your job is to:
1. Understand user queries
2. Coordinate between the retriever and analyzer agents
3. Ensure comprehensive answers are provided

Workflow:
1. First, ask the DocumentRetriever to search for relevant documents
2. Then, ask the ContentAnalyzer to analyze the retrieved documents and answer the query
3. Provide the final answer to the user

Always ensure the answer is based on the retrieved documents."""
        
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
        )


class AgenticRAGSystem:
    """Main Agentic RAG system coordinating all agents."""
    
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        llm_config: Dict,
        max_round: int = 10
    ):
        """
        Initialize Agentic RAG system.
        
        Args:
            vectorstore: Vector store manager instance
            llm_config: LLM configuration
            max_round: Maximum number of conversation rounds
        """
        self.vectorstore = vectorstore
        self.llm_config = llm_config
        self.max_round = max_round
        
        # Initialize agents
        self.retriever = RetrieverAgent(vectorstore, llm_config)
        self.analyzer = AnalyzerAgent(llm_config)
        self.orchestrator = OrchestratorAgent(llm_config)
        
        # Create user proxy agent
        self.user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
        )
        
        # Register search function with retriever agent
        self.register_functions()
    
    def register_functions(self):
        """Register custom functions with agents."""
        
        @self.user_proxy.register_for_execution()
        @self.retriever.agent.register_for_llm(description="Search for relevant documents in the vector database")
        def search_documents(query: str, n_results: int = 5) -> str:
            """Search for relevant documents."""
            return self.retriever.search_documents(query, n_results)
    
    def query(self, question: str) -> str:
        """
        Process a query through the agentic RAG system.
        
        Args:
            question: User question
            
        Returns:
            Answer from the system
        """
        # Create group chat
        groupchat = GroupChat(
            agents=[
                self.user_proxy,
                self.orchestrator.agent,
                self.retriever.agent,
                self.analyzer.agent
            ],
            messages=[],
            max_round=self.max_round,
        )
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )
        
        # Initiate chat
        self.user_proxy.initiate_chat(
            manager,
            message=f"Please answer this question: {question}"
        )
        
        # Get the last message as response
        if groupchat.messages:
            return groupchat.messages[-1]['content']
        return "No response generated."
