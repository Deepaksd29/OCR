import chromadb
from chromadb.config import Settings
import uuid
from typing import Dict, Any, List
from utlity.env_load import env_data



class ChromaDBManager:

    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.CloudClient(
                    api_key=env_data.CHROMA_API_KEY,
                    tenant=env_data.CHROMA_TENANT,
                    database=env_data.CHROMA_DATABASE
                    )

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, document_data: Dict) -> str:
        doc_id = str(uuid.uuid4())

        chunks = self.split_text(document_data["text"])
        
        if "tables" in document_data:
            for table in document_data["tables"]:
                table_chunks = self.split_text(table["csv_data"], prefix="TABLE: ")
                chunks.extend(table_chunks)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            metadata = {
                "filename": document_data["filename"],
                "mime_type": document_data["mime_type"],
                "processing_method": document_data["processing_method"],
                "timestamp": document_data["timestamp"],
                "chunk_index": i,
                "parent_doc_id": doc_id,
                "page_count": document_data.get("page_count", 1),
                "has_tables": document_data.get("has_tables", False),
                "has_images": document_data.get("has_images", False)
            }
            
            self.collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[chunk_id]
            )
        
        return doc_id
    
    def split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200, prefix: str = ""):
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if prefix:
                chunk = prefix + chunk
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        return results
    
    def get_document_stats(self) -> Dict[str, Any]:
        all_docs = self.get_all_documents()
        
        if not all_docs['metadatas']:
            return {"total_chunks": 0, "unique_documents": 0, "file_types": {}}
        
        unique_docs = set()
        file_types = {}
        
        for metadata in all_docs['metadatas']:
            unique_docs.add(metadata['parent_doc_id'])
            mime_type = metadata.get('mime_type', 'unknown')
            file_types[mime_type] = file_types.get(mime_type, 0) + 1
        
        return {
            "total_chunks": len(all_docs['metadatas']),
            "unique_documents": len(unique_docs),
            "file_types": file_types
        }
        
    
    def clear_collection(self, collection_name):
        try:
            self.client.delete_collection(name=collection_name)
        except:
            import traceback
            traceback.print_exc()