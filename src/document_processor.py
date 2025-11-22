"""
Document processing utilities for the Agentic RAG system.
"""
import os
from typing import List, Optional
from pathlib import Path
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process various document formats for RAG ingestion."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def read_pdf(self, file_path: str) -> str:
        """Read text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def read_docx(self, file_path: str) -> str:
        """Read text from DOCX file."""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def read_txt(self, file_path: str) -> str:
        """Read text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def process_document(self, file_path: str) -> List[str]:
        """
        Process a document and split it into chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of text chunks
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text = self.read_pdf(file_path)
        elif file_ext == '.docx':
            text = self.read_docx(file_path)
        elif file_ext == '.txt':
            text = self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def process_directory(self, directory_path: str) -> List[dict]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        documents = []
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in supported_extensions:
                    try:
                        chunks = self.process_document(file_path)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                'text': chunk,
                                'metadata': {
                                    'source': file_path,
                                    'chunk_id': i,
                                    'file_name': file
                                }
                            })
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        return documents
