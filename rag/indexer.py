"""
AI Flow - RAG (Retrieval Augmented Generation) System
Indexes existing codebase for context-aware code generation
"""

import os
from pathlib import Path
from typing import List, Optional
import hashlib

import google.generativeai as genai
from pydantic import BaseModel


class CodeChunk(BaseModel):
    """A chunk of code with metadata"""
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_hash: str
    embedding: Optional[List[float]] = None


class CodebaseIndexer:
    """
    Indexes a codebase for RAG-based code generation.
    Uses Gemini's embedding API (free tier).
    """
    
    SUPPORTED_EXTENSIONS = {
        '.ts', '.tsx', '.js', '.jsx',  # JavaScript/TypeScript
        '.py',  # Python
        '.prisma',  # Prisma schema
        '.sql',  # SQL
        '.json',  # JSON configs
        '.yaml', '.yml',  # YAML configs
        '.md',  # Markdown docs
    }
    
    IGNORE_DIRS = {
        'node_modules', '.git', 'dist', 'build', '.next',
        '__pycache__', '.pytest_cache', 'coverage',
        '.turbo', '.vercel', '.cache',
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[CodeChunk] = []
    
    def _should_index_file(self, path: Path) -> bool:
        """Check if a file should be indexed"""
        # Check extension
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            return False
        
        # Check if in ignored directory
        for part in path.parts:
            if part in self.IGNORE_DIRS:
                return False
        
        return True
    
    def _chunk_content(self, content: str, file_path: str) -> List[CodeChunk]:
        """Split content into chunks with overlap"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_start = 1
        current_size = 0
        
        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_start,
                    end_line=i - 1,
                    chunk_hash=chunk_hash,
                ))
                
                # Start new chunk with overlap
                overlap_lines = max(0, len(current_chunk) - self.chunk_overlap // 50)
                current_chunk = current_chunk[overlap_lines:]
                current_start = i - len(current_chunk)
                current_size = sum(len(l) + 1 for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_hash=chunk_hash,
            ))
        
        return chunks
    
    def index_directory(self, directory: str) -> int:
        """Index all supported files in a directory"""
        dir_path = Path(directory)
        indexed_count = 0
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and self._should_index_file(file_path):
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    relative_path = str(file_path.relative_to(dir_path))
                    
                    file_chunks = self._chunk_content(content, relative_path)
                    self.chunks.extend(file_chunks)
                    indexed_count += 1
                    
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")
        
        return indexed_count
    
    async def generate_embeddings(self) -> None:
        """Generate embeddings for all chunks using Gemini"""
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        
        for i, chunk in enumerate(self.chunks):
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=f"File: {chunk.file_path}\n{chunk.content}",
                    task_type="retrieval_document",
                )
                chunk.embedding = result['embedding']
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.chunks)} chunks")
                    
            except Exception as e:
                print(f"Error generating embedding for chunk: {e}")
        
        print("Embedding generation complete")
    
    def search(self, query: str, top_k: int = 5) -> List[CodeChunk]:
        """Search for relevant code chunks"""
        # Generate query embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query",
        )
        query_embedding = result['embedding']
        
        # Calculate cosine similarity
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x ** 2 for x in a) ** 0.5
            norm_b = sum(x ** 2 for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
        
        # Score all chunks
        scored_chunks = []
        for chunk in self.chunks:
            if chunk.embedding:
                score = cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append((score, chunk))
        
        # Return top-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def get_context_for_task(self, task_description: str, top_k: int = 5) -> str:
        """Get relevant code context for a task"""
        relevant_chunks = self.search(task_description, top_k)
        
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(
                f"--- {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}) ---\n"
                f"{chunk.content}\n"
            )
        
        return "\n".join(context_parts)


class DocumentIndexer:
    """
    Indexes documentation files for requirements and context.
    Specifically designed for the /docs directory.
    """
    
    def __init__(self):
        self.documents: dict = {}  # {filename: content}
    
    def index_docs_directory(self, docs_dir: str) -> int:
        """Index all markdown files in the docs directory"""
        docs_path = Path(docs_dir)
        indexed_count = 0
        
        for file_path in docs_path.glob('*.md'):
            try:
                content = file_path.read_text(encoding='utf-8')
                self.documents[file_path.name] = content
                indexed_count += 1
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        return indexed_count
    
    def get_all_docs_content(self) -> str:
        """Get combined content of all documents"""
        parts = []
        for filename, content in sorted(self.documents.items()):
            parts.append(f"=== {filename} ===\n{content}\n")
        return "\n".join(parts)
    
    def get_doc_by_type(self, doc_type: str) -> Optional[str]:
        """Get document by type (BRD, TAD, FSD, etc.)"""
        for filename, content in self.documents.items():
            if doc_type.lower() in filename.lower():
                return content
        return None
