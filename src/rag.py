from typing import Dict, List, Any
import torch
from chromadb import Client, Settings
from eval import QAEvaluator
import requests

class RAGPipeline:
    def __init__(self, qa_model_name="microsoft/deberta-v3-base"):
        self.QA_API_URL = f"https://api-inference.huggingface.co/models/{qa_model_name}"
        self.EMBED_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": "Bearer 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"}
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        try:
            self.chroma_client.delete_collection("document_store")
        except:
            pass
            
        self.collection = self.chroma_client.create_collection(
            name="document_store",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embeddings(self, text: str) -> List[float]:
        try:
            response = requests.post(
                self.EMBED_API_URL,
                headers=self.headers,
                json={"inputs": text}
            )
            return response.json()
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return []

    def chunk_context(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            if end > len(words):
                end = len(words)
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def add_documents(self, documents: List[str], ids: List[str] = None):
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
        
        all_chunks = []
        chunk_ids = []
        for idx, doc in enumerate(documents):
            chunks = self.chunk_context(doc)
            all_chunks.extend(chunks)
            chunk_ids.extend([f"{ids[idx]}_{i}" for i in range(len(chunks))])
            
        embeddings = [self._get_embeddings(chunk) for chunk in all_chunks]
        
        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            ids=chunk_ids
        )

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        query_embedding = self._get_embeddings(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        threshold = 0.7
        relevant_docs = []
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            if distance < threshold:
                relevant_docs.append(doc)
        
        return relevant_docs if relevant_docs else results['documents'][0][:1]

    def _query_api(self, question: str, context: str) -> str:
        try:
            payload = {
                "inputs": {
                    "question": question,
                    "context": context
                }
            }
            response = requests.post(self.QA_API_URL, headers=self.headers, json=payload)
            output = response.json()
            return output.get('answer', '')
        except Exception as e:
            print(f"Error in API call: {e}")
            return ""

    def __call__(self, question: str, context: str = None) -> str:
        try:
            retrieved_docs = self.retrieve(question)
            if not retrieved_docs:
                return "No relevant information found"
            context = " ".join(retrieved_docs)
            
            answer = self._query_api(question, context)
            return answer.strip() if answer else "No answer found"
                
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return ""

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Load and add context document
with open('data/context.txt', 'r') as f:
    context = f.read()
    
# Add the context as a document
rag_pipeline.add_documents([context])

# Initialize evaluator
rag_evaluator = QAEvaluator('out/rag_results.json')

# Evaluate pipeline
rag_evaluator.evaluate_pipeline(
    pipeline=rag_pipeline,
    questions_file='data/questions.json',
    context_file='data/context.txt'
)
