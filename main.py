import streamlit as st
import os
from utlity.llm import DocumentQASystem
import tempfile
from utlity.env_load import env_data
from uuid import uuid4

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    st.set_page_config(
        page_title="Document QA System with Docling",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document QA System")
    st.markdown("Upload documents and ask questions")
    

    
    # Sidebar for configuration
    with st.sidebar:
        if st.button("Clear"):
            st.session_state.qa_system.clear_system(f"col{st.session_state.session_id}")
            st.session_state.clear()
            

        if "qa_system" not in st.session_state:
                st.session_state.qa_system = None
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid4())
        api_key = env_data.GOOGLE_API_KEY
        system_inti = DocumentQASystem(api_key, collection_name=st.session_state.session_id)
        st.session_state.qa_system = system_inti
        
        st.divider()
        
        
        
        # System stats
        if st.session_state.qa_system:
            st.header("📊 System Statistics")
            stats = st.session_state.qa_system.get_system_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["unique_documents"])
            with col2:
                st.metric("Text Chunks", stats["total_chunks"])
            
            if stats["file_types"]:
                st.write("**File Types:**")
                for file_type, count in stats["file_types"].items():
                    st.write(f"• {file_type}: {count}")
                    
            
        st.divider()
        
        # Document upload section
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'png', 'jpg', 'jpeg']
        )
        
        if uploaded_files and st.session_state.qa_system:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name} with Docling..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:    
                            result = st.session_state.qa_system.process_and_store_document(tmp_path)
                            
                            if result["success"]:
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                
                                # Show processing details
                                with st.expander("Processing Details"):
                                    metadata = result["metadata"]
                                    st.write(f"**Word Count:** {metadata.get('word_count', 0)}")
                                    st.write(f"**Pages:** {metadata.get('page_count', 1)}")
                                    st.write(f"**Has Tables:** {'Yes' if metadata.get('has_tables') else 'No'}")
                                    st.write(f"**Has Images:** {'Yes' if metadata.get('has_images') else 'No'}")
                                    st.write(f"**Processing Method:** {metadata.get('processing_method', 'Unknown')}")
                            else:
                                st.error(f"❌ Error processing {uploaded_file.name}: {result['error']}")
                        
                        finally:
                            os.unlink(tmp_path)
    

    if st.session_state.qa_system:
        st.header("💬 Ask Questions")
        

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                

                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources & Details"):
                        for source in message["sources"]:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📄 **{source['filename']}**")
                                features = []
                                if source.get('has_tables'):
                                    features.append("Tables")
                                if source.get('has_images'):
                                    features.append("Images")
                                if features:
                                    st.write(f"   Contains: {', '.join(features)}")
                            with col2:
                                if 'relevance_score' in source:
                                    st.metric("Relevance", f"{source['relevance_score']:.2f}")
        

        processing = False
        if uploaded_files and st.session_state.qa_system:
            for uploaded_file in uploaded_files:
                if st.session_state.get(f"processing_{uploaded_file.name}", False):
                    processing = True
                    break


        if 'doc_processing' not in st.session_state:
            st.session_state['doc_processing'] = False


        st.session_state['doc_processing'] = processing

        chat_disabled = st.session_state['doc_processing']

        prompt = st.chat_input("Ask a question about your documents...", disabled=chat_disabled)
        if chat_disabled:
            st.info("Please wait until all documents are processed before asking a question.")

        if prompt and not chat_disabled:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents and generating answer..."):
                    response = st.session_state.qa_system.answer_question(prompt)
                st.markdown(response["answer"])

                if response.get("confidence", 0) > 0:
                    confidence_color = "green" if response["confidence"] > 0.7 else "orange" if response["confidence"] > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{response['confidence']:.2f}]")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response.get("sources", [])
            })
    
   

if __name__ == "__main__":
    main()