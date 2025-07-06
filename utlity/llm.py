
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from utlity.chromadb import ChromaDBManager
from utlity.documnet_proesser import DocumentProcessor






class GeminiQAAgent:
    
    def __init__(self, api_key: str):

      
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            You are an intelligent document analysis assistant. 
            Based on the provided context from documents, answer the question accurately and comprehensively.
            Context from documents:
            {context}
            Question: {question}
            Instructions:
            1.Provide a detailed, accurate answer based on the context 
            2.If the answer requires information not in the context, clearly state what's missing
            3.Use specific information and quotes from the documents when relevant
            4.If you find tables or structured data, present it clearly
            5.If the question involves calculations or analysis, show your reasoning
            6.Maintain a helpful and professional tone\n- If there are multiple perspectives in the documents, present them fairly
            {source_info}"""
        )
    def generate_answer(self, question: str, context: str, metadata: List[Dict] = None) -> str:
        """Generate answer using Gemini via LangChain with context and metadata"""
        # Prepare metadata information
        source_info = ""
        if metadata:
            sources = []
            for meta in metadata:
                source_detail = f"- {meta.get('filename', 'Unknown file')}"
                if meta.get('has_tables'):
                    source_detail += " (contains tables)"
                if meta.get('has_images'):
                    source_detail += " (contains images)"
                sources.append(source_detail)
            source_info = f"\n\nSources:\n" + "\n".join(set(sources))

        prompt = self.prompt_template.format_messages(
            context=context,
            question=question,
            source_info=source_info
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"Sorry, I encountered an error while generating the answer: {e}"

class DocumentQASystem:

    
    def __init__(self, gemini_api_key: str, collection_name:str):
        self.processor = DocumentProcessor()
        self.db_manager = ChromaDBManager(collection_name=f"col{collection_name}")
        self.qa_agent = GeminiQAAgent(gemini_api_key)
    
    def process_and_store_document(self, filepath: str):
        try:
            # Extract text and metadata using Docling
            document_data = self.processor.extract_text_from_file(filepath)
            
            # Store in ChromaDB
            doc_id = self.db_manager.add_document(document_data)
            
            return {
                "success": True,
                "doc_id": doc_id,
                "metadata": document_data
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def answer_question(self, question: str):
        try:
            search_results = self.db_manager.search_documents(question, n_results=8)
            
            if not search_results["documents"]:
                return {
                    "answer": "I don't have any relevant documents to answer your question. Please upload some documents first.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            context = "\n\n".join(search_results["documents"][0])
            
            answer = self.qa_agent.generate_answer(question, context, search_results["metadatas"][0])

            sources = []
            confidence_scores = []
            
            if search_results["metadatas"]:
                for i, metadata in enumerate(search_results["metadatas"][0]):
                    source_info = {
                        "filename": metadata.get("filename", "Unknown"),
                        "processing_method": metadata.get("processing_method", "Unknown"),
                        "has_tables": metadata.get("has_tables", False),
                        "has_images": metadata.get("has_images", False),
                        "page_count": metadata.get("page_count", 1)
                    }
                    
                    if search_results.get("distances") and i < len(search_results["distances"][0]):
                        source_info["relevance_score"] = 1 - search_results["distances"][0][i]
                        confidence_scores.append(source_info["relevance_score"])
                    
                    sources.append(source_info)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "confidence": avg_confidence
            }
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {e}",
                "sources": [],
                "confidence": 0.0
            }
    
    def get_system_stats(self):
        return self.db_manager.get_document_stats()
    
    def clear_system(self, collection_name:str):
        self.db_manager.clear_collection(collection_name=collection_name)
