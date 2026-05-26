import streamlit as st
import os
import json
from src.pipeline import streamlit_single_click_pipeline_sync, query_documents_sync

def query_documents():
    """Query the processed documents using RAG pipeline."""
    try:
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not pinecone_api_key:
            st.error("❌ Pinecone API key not found! Please set PINECONE_API_KEY in secrets or environment.")
            return
            
        if not gemini_api_key:
            st.warning("⚠️ Gemini API key not found! Using fallback mode with limited accuracy.")
            gemini_api_key = None  # Use None for fallback mode
        
        # Show API status
        if gemini_api_key:
            st.success("✅ LLM Ready: Advanced analysis available")
        else:
            st.warning("⚠️ LLM Unavailable: Using fallback mode with reduced accuracy")
        
        # Sample queries
        with st.expander("💡 Sample Queries", expanded=False):
            st.markdown("""
            **Coverage Questions:**
            - "What is covered under accidental death benefit?"
            - "Are pre-existing conditions covered?"
            - "What is the waiting period for maternity benefits?"
            - "Is dental surgery covered for a 35-year-old female?"
            - "Does the policy cover mental health treatment?"
            
            **Claim Questions:**
            - "How do I file a claim for medical expenses?"
            - "What documents are needed for death claim?"
            - "What is the claim settlement process?"
            - "Can I submit claims online?"
            - "What is the maximum claim amount per year?"
            
            **Policy Questions:**
            - "What is the premium payment term?"
            - "Can I surrender my policy early?"
            - "What are the tax benefits available?"
            - "Is there coverage for 3 months old policy?"
            - "What happens if I miss premium payments?"
            - "Are there any age restrictions for coverage?"
            """)
            
            # Quick action buttons for common queries
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🦷 Dental Surgery Coverage", use_container_width=True, key="dental_btn"):
                    st.session_state['query_input'] = "Is dental surgery covered for a 35-year-old female?"
                    st.rerun()
                if st.button("🤰 Maternity Benefits", use_container_width=True, key="maternity_btn"):
                    st.session_state['query_input'] = "What is the waiting period for maternity benefits?"
                    st.rerun()
            with col2:
                if st.button("💰 Claim Process", use_container_width=True, key="claim_btn"):
                    st.session_state['query_input'] = "How do I file a claim for medical expenses?"
                    st.rerun()
                if st.button("📋 Policy Duration", use_container_width=True, key="policy_btn"):
                    st.session_state['query_input'] = "Is there coverage for 3 months old policy?"
                    st.rerun()
        
        # Initialize session state for query input persistence
        if 'persistent_query' not in st.session_state:
            st.session_state.persistent_query = ''
        
        # Handle button clicks for sample queries
        if 'query_input' in st.session_state:
            st.session_state.persistent_query = st.session_state['query_input']
            del st.session_state['query_input']
        
        # Main query input with form for Enter key submission
        with st.form("query_form", clear_on_submit=False):
            user_query = st.text_area(
                "Enter your question:",
                value=st.session_state.persistent_query,
                placeholder="e.g., Is dental surgery covered for a 35-year-old female?",
                height=100,
                key="query_text_input"
            )
            
            # Create columns for submit and clear buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submitted = st.form_submit_button("🔍 Search Documents", type="primary")
            with col2:
                clear_query = st.form_submit_button("🗑️ Clear Query")
            with col3:
                clear_results = st.form_submit_button("🧹 Clear Results")
        
        # Handle clear query button
        if clear_query:
            st.session_state.persistent_query = ''
            if 'query_results' in st.session_state:
                del st.session_state.query_results
            st.rerun()
        
        # Handle clear results button
        if clear_results:
            if 'query_results' in st.session_state:
                del st.session_state.query_results
            st.rerun()
        
        # Process query if search button is clicked
        if submitted and user_query and user_query.strip():
            # Save the query to session state for persistence
            st.session_state.persistent_query = user_query.strip()
            
            with st.spinner("🔍 Searching documents and analyzing..."):
                try:
                    # Process query using pipeline function
                    result = query_documents_sync(
                        query=user_query.strip(),
                        pinecone_api_key=pinecone_api_key,
                        gemini_api_key=gemini_api_key,
                        index_name="policy-index"
                    )
                    
                    # Store results in session state
                    st.session_state.query_results = {
                        'query': user_query.strip(),
                        'result': result
                    }
                    
                except Exception as e:
                    st.error(f"❌ Query processing failed: {str(e)}")
                    st.session_state.query_results = {
                        'query': user_query.strip(),
                        'result': {
                            'success': False,
                            'error': str(e)
                        }
                    }
                
                # Rerun to display results
                st.rerun()
        
        # Display results from session state (persistent across reruns)
        if st.session_state.query_results:
            query = st.session_state.query_results['query']
            result = st.session_state.query_results['result']
            
            st.markdown("---")
            st.markdown(f"### 📊 Results for: *\"{query}\"*")
            
            if result.get("success"):
                # Show evaluation
                evaluation = result.get("evaluation", {})
                decision = evaluation.get("decision", "unknown")
                confidence = evaluation.get("confidence", 0.0)
                
                # Color-coded decision
                if decision == "covered":
                    st.success(f"✅ **COVERED** (Confidence: {confidence:.1%})")
                elif decision == "not_covered":
                    st.error(f"❌ **NOT COVERED** (Confidence: {confidence:.1%})")
                elif decision == "partial":
                    st.warning(f"⚠️ **PARTIALLY COVERED** (Confidence: {confidence:.1%})")
                else:
                    st.info(f"🤔 **UNCLEAR** (Confidence: {confidence:.1%})")
                
                # Show answer from LLM if available
                answer = evaluation.get("answer")
                if answer:
                    st.markdown("### 💬 Answer")
                    st.markdown(f"_{answer}_")
                
                # Show justification
                justification = evaluation.get("justification", "No explanation available")
                st.markdown(f"**Explanation:** {justification}")
                
                # Show relevant clauses
                relevant_clauses = evaluation.get("relevant_clauses", [])
                if relevant_clauses:
                    st.markdown("### 📄 Relevant Policy Clauses")
                    for i, clause in enumerate(relevant_clauses, 1):
                        # Handle both string and dict formats
                        if isinstance(clause, dict):
                            section = clause.get('section', 'Unknown Section')
                            content = clause.get('content', 'No content available')
                            page = clause.get('page')
                            
                            with st.expander(f"Clause {i}: {section}"):
                                st.markdown(f"**Content:** {content}")
                                if page:
                                    st.markdown(f"**Source:** Page {page}")
                        else:
                            # Handle string format (fallback)
                            with st.expander(f"Clause {i}: Policy Clause"):
                                st.markdown(f"**Content:** {clause}")
                
                # Show search results
                search_results = result.get("search_results", [])
                if search_results:
                    st.markdown("### 🔍 Source Documents")
                    for i, chunk in enumerate(search_results, 1):
                        score = chunk.get('score', 0.0)
                        source = chunk.get('source_document', 'Unknown')
                        content = chunk.get('text', 'No content available')
                        
                        with st.expander(f"Result {i}: {source} (Score: {score:.3f})"):
                            st.markdown(content[:1000] + "..." if len(content) > 1000 else content)
                
                # Show extracted entities
                entities = result.get("entities", {})
                if entities and any(entities.values()):
                    with st.expander("🏷️ Extracted Information", expanded=False):
                        st.json(entities)
                
                # Show technical details
                with st.expander("🔧 Technical Details", expanded=False):
                    api_status = result.get("api_status", {})
                    tech_info = {
                        "Search Method": evaluation.get("search_method", "unknown"),
                        "Chunks Used": evaluation.get("final_chunks_used", 0),
                        "LLM Available": api_status.get('gemini_available', False),
                        "Model": api_status.get('model_name', 'N/A')
                    }
                    st.json(tech_info)
            
            else:
                st.error("❌ Query processing failed!")
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
    except Exception as e:
        st.error(f"❌ Query system error: {str(e)}")
        st.exception(e)


def single_click_pipeline():
    """Single-click pipeline for complete document processing."""
    try:
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            return {"success": False, "error": "Pinecone API key not found!"}
        
        # Run the complete pipeline
        result = streamlit_single_click_pipeline_sync(
            pinecone_api_key=pinecone_api_key,
            force_reprocess=False
        )
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    st.set_page_config(
        page_title="Insurance Policy RAG System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state for processing status
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}
    if 'query_results' not in st.session_state:
        st.session_state.query_results = None
    if 'persistent_query' not in st.session_state:
        st.session_state.persistent_query = ''
    
    st.title("🏥 Insurance Policy RAG System")
    st.markdown("**AI-Powered Insurance Policy Analysis** • Ask questions and get instant answers with source citations")
    st.markdown("---")
    
    # Sidebar for document processing
    with st.sidebar:
        st.header("📄 Document Processing")
        st.markdown("Process PDF documents to build the knowledge base")
        
        # Show current document status
        docs_dir = "docs"
        if os.path.exists(docs_dir):
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            if pdf_files:
                st.success(f"📁 {len(pdf_files)} PDF files found")
                with st.expander("📋 Files in docs/", expanded=False):
                    for pdf in pdf_files:
                        st.markdown(f"• {pdf}")
            else:
                st.warning("📂 No PDF files found")
                st.info("Place PDF files in the `docs/` folder")
        else:
            st.error("📂 docs/ directory not found")
        
        st.markdown("---")
        
        # Processing button
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("🔄 Processing documents..."):
                result = single_click_pipeline()
                
                if result.get("success", False):
                    # Store processing results in session state
                    st.session_state.processing_complete = True
                    st.session_state.processing_stats = result.get("statistics", {})
                    
                    # Safely get statistics and timing with defaults
                    stats = result.get("statistics", {})
                    timing = result.get("timing", {})
                    
                    processed_files = stats.get("processed_files", 0)
                    skipped_files = stats.get("skipped_files", 0)
                    
                    if processed_files > 0:
                        st.success("✅ Processing Complete!")
                        # Display compact statistics
                        st.metric("Files Processed", processed_files)
                        st.metric("Chunks Created", stats.get("total_chunks", 0)) 
                        st.metric("Vectors Indexed", stats.get("indexed_vectors", 0))
                        
                        # Show timing info
                        total_time = timing.get('total_time', 'N/A')
                        st.info(f"⏱️ Total: {total_time}")
                    else:
                        st.info("✅ All documents already processed!")
                        st.info(f"📋 {skipped_files} files were already up-to-date")
                        st.info("💡 Use 'Force Reprocess All' to reprocess existing documents")
                        
                        # Still show basic stats
                        st.metric("Total Chunks", stats.get("total_chunks", 0))
                        st.metric("Total Vectors", stats.get("indexed_vectors", 0))
                        
                else:
                    st.error("❌ Processing Failed")
                    error_msg = result.get('error', 'Unknown error occurred')
                    step = result.get('step', 'unknown')
                    st.error(f"Error in {step}: {error_msg}")
                    
                    # Show additional debug information if available
                    if 'user_message' in result:
                        st.info(result['user_message'])
        
        # Show persistent processing status
        elif st.session_state.processing_complete:
            st.success("✅ All documents already processed!")
            stats = st.session_state.processing_stats
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Total Vectors", stats.get("indexed_vectors", 0))
            st.info("💡 Use 'Force Reprocess All' to reprocess existing documents")
        
        # Force reprocess option
        st.markdown("---")
        if st.button("🔄 Force Reprocess All", use_container_width=True):
            with st.spinner("🔄 Reprocessing all documents..."):
                # Get API keys from secrets or environment
                pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
                
                if pinecone_api_key:
                    result = streamlit_single_click_pipeline_sync(
                        pinecone_api_key=pinecone_api_key,
                        force_reprocess=True
                    )
                    
                    if result.get("success", False):
                        # Update session state
                        st.session_state.processing_complete = True
                        st.session_state.processing_stats = result.get("statistics", {})
                        
                        st.success("✅ Reprocessing Complete!")
                        stats = result.get("statistics", {})
                        st.metric("Total Processed", stats.get("processed_files", 0))
                    else:
                        st.error("❌ Reprocessing Failed")
                        st.error(result.get('error', 'Unknown error'))
                else:
                    st.error("❌ API key not found")
    
    # Main content area for queries
    st.markdown("### 🔍 Ask Questions About Your Insurance Policies")
    
    # Query interface (now taking full width)
    query_documents()

if __name__ == "__main__":
    main()
