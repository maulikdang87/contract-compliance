import streamlit as st
from modules.document_processor import DocumentProcessor
from modules.agents.base_agent import BaseContractAgent
from modules.vector_store import VectorStore
from modules.compliance_engine import ComplianceEngine
from config.settings import settings
import os
from dotenv import load_dotenv

# Create necessary directories
settings.create_directories()
load_dotenv()

def main():
    st.title("🔍 Enhanced Contract Compliance Checker")
    st.markdown("Upload a contract document for comprehensive compliance analysis")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    compliance_engine = ComplianceEngine()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a contract file",
        type=['pdf', 'docx', 'txt'],
        help="Upload a PDF, Word document, or text file"
    )
    
    if uploaded_file is not None:
        # Save and process file
        file_path = os.path.join(settings.CONTRACTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        with st.spinner("Processing document..."):
            try:
                # Extract and process
                text = doc_processor.extract_text(file_path)
                metadata = doc_processor.extract_metadata(text)
                chunks = doc_processor.chunk_document(text)
                vector_store.add_documents(chunks)
                
                # FIXED: Initialize agent with dependencies AFTER processing
                agent = BaseContractAgent(
                    vector_store=vector_store,
                    compliance_engine=compliance_engine
                )
                
                # Display basic info
                st.subheader("📋 Document Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Word Count", metadata['word_count'])
                with col2:
                    st.metric("Contract Type", metadata['contract_type'].title())
                with col3:
                    st.metric("Parties Found", len(metadata['parties']))
                
                # Show extracted parties
                if metadata['parties']:
                    st.write("**Parties:**", ", ".join(metadata['parties']))
                
                # Compliance Analysis
                st.subheader("🔍 Compliance Analysis")
                issues = compliance_engine.check_compliance(text)
                risk_assessment = compliance_engine.calculate_risk_score(issues)
                
                # Risk Score Display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{risk_assessment['score']}/100")
                with col2:
                    st.metric("Risk Level", risk_assessment['level'].title())
                with col3:
                    st.metric("Issues Found", risk_assessment['total_issues'])
                
                # Detailed Issues
                if issues:
                    st.subheader("⚠️ Compliance Issues")
                    for issue in issues:
                        with st.expander(f"{issue.rule_name} ({issue.severity} severity)"):
                            st.write(f"**Issue:** {issue.violation_message}")
                            st.write(f"**Missing Elements:** {', '.join(issue.missing_elements)}")
                            st.write(f"**Confidence:** {issue.confidence_score:.2f}")
                else:
                    st.success("✅ No compliance issues found!")
                
                # Semantic Search Demo
                st.subheader("🔍 Semantic Search Test")
                search_query = st.text_input("Search for specific clauses:", value="termination clause")
                
                if search_query:
                    search_results = vector_store.search_similar(search_query, k=3)
                    if search_results:
                        st.write(f"**Top relevant sections for '{search_query}':**")
                        for i, chunk in enumerate(search_results, 1):
                            with st.expander(f"Result {i}"):
                                st.text_area("Content", chunk[:500] + "..." if len(chunk) > 500 else chunk, height=100)
                    else:
                        st.write("No relevant sections found.")
                
                # Interactive Analysis
                st.subheader("🤖 Ask the AI Agent")
                user_query = st.text_input("Ask about the contract (e.g., 'Check termination clauses')")
                
                if user_query:
                    with st.spinner("Agent analyzing..."):
                        # FIXED: Use the correct method with proper parameters
                        response = agent.analyze_contract_comprehensive(user_query, text)
                        st.markdown(response)
                
                # Show document chunks (for debugging)
                with st.expander("📄 Document Chunks (Debug)"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.text_area(f"Chunk {i+1}", chunk, height=100)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
