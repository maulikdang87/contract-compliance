import streamlit as st
from modules.document_processor import DocumentProcessor
from modules.agents.enhanced_base_agent import EnhancedBaseAgent
from modules.vector_store import VectorStore
from modules.compliance_engine import ComplianceEngine
from config.settings import settings
import os
from dotenv import load_dotenv
import re

# Create necessary directories
settings.create_directories()
load_dotenv()

def main():
    st.set_page_config(
        page_title="Enhanced Contract Compliance Checker",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header with improved visual hierarchy
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0;">🔍 Enhanced Contract Compliance Checker</h1>
        <p style="color: #666; font-size: 1.1rem;">AI-powered contract analysis with Gemini embedding agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced sidebar with better organization
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            st.markdown("#### 🤖 Model Settings")
            model_choice = st.selectbox(
                "AI Model",
                ["gemini-1.5-flash"],
                index=0
            )
        
        st.markdown("#### 🔧 Agent Configuration")
        with st.container():
            max_iterations = st.slider("Max Agent Iterations", 1, 10, 5)
            search_depth = st.slider("Search Depth", 1, 10, 3)
        
        st.markdown("#### 📊 Analysis Settings")
        with st.container():
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Comprehensive"],
                index=0
            )
        
        st.markdown("---")
        
        # Add system information in sidebar
        st.markdown("#### 📱 System Info")
        st.info("💡 **Tip**: Upload PDF, Word, or text files for analysis")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    compliance_engine = ComplianceEngine()
    
    # Main content area with improved layout
    main_col, status_col = st.columns([2.5, 1])
    
    with main_col:
        # Document upload section with better visual appeal
        st.markdown("## 📄 Document Upload")
        
        upload_container = st.container()
        with upload_container:
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
            
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            
            # Processing with enhanced agents
            with st.spinner("🤖 Processing document with Gemini agents..."):
                try:
                    # Extract and process
                    text = doc_processor.extract_text(file_path)
                    metadata = doc_processor.extract_metadata(text)
                    
                    # Basic chunking and vector storage
                    chunks = doc_processor.chunk_document(text)
                    vector_store.add_documents(chunks)
                    
                    # Initialize enhanced agent with Gemini capabilities
                    agent = EnhancedBaseAgent(
                        model_name=model_choice,
                        vector_store=vector_store,
                        compliance_engine=compliance_engine,
                        max_iterations=max_iterations
                    )
                    
                    # Document Information Dashboard with improved spacing
                    st.markdown("## 📊 Document Analysis Dashboard")
                    
                    # Metrics row with better visual structure
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("📝 Word Count", f"{metadata['word_count']:,}")
                    with metric_cols[1]:
                        st.metric("📋 Contract Type", metadata['contract_type'].title())
                    with metric_cols[2]:
                        st.metric("👥 Parties Found", len(metadata['parties']))
                    with metric_cols[3]:
                        st.metric("📑 Sections", len(chunks))
                    
                    st.markdown("---")
                    
                    # Enhanced Compliance Analysis with Gemini
                    st.markdown("## 🔍 AI-Powered Compliance Analysis")
                    
                    # Use Gemini agent for compliance analysis
                    with st.spinner("🤖 Analyzing compliance with Gemini AI..."):
                        gemini_compliance_results = agent.compliance_agent.analyze_compliance_intelligently(
                            text, "employment_contracts"
                        )
                    
                    # Fallback to basic compliance engine if Gemini fails
                    if 'error' in gemini_compliance_results:
                        st.warning("⚠️ Gemini analysis unavailable, using basic compliance engine")
                        issues = compliance_engine.check_compliance(text)
                        risk_assessment = compliance_engine.calculate_risk_score(issues)
                        
                        compliance_results = {
                            'issues': issues,
                            'risk_assessment': risk_assessment,
                            'summary': f"Analysis complete. Found {len(issues)} issues with {risk_assessment['level']} risk level."
                        }
                    else:
                        # Use Gemini results
                        compliance_results = {
                            'issues': gemini_compliance_results.get('compliance_issues', []),
                            'risk_assessment': gemini_compliance_results.get('risk_assessment', {}),
                            'summary': gemini_compliance_results.get('summary', 'Gemini analysis completed.')
                        }
                    
                    # Display results in organized tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📋 Overview", 
                        "🚨 Issues", 
                        "🔍 Gemini Smart Search", 
                        "📊 Risk Assessment"
                    ])
                    
                    with tab1:
                        st.markdown("### 📄 Contract Overview")
                        
                        # Better organization of overview information
                        overview_col1, overview_col2 = st.columns([1, 1])
                        
                        with overview_col1:
                            if metadata['parties']:
                                st.markdown("**🤝 Contract Parties:**")
                                for party in metadata['parties']:
                                    st.write(f"• {party}")
                            else:
                                st.info("No parties identified")
                        
                        with overview_col2:
                            if metadata.get('dates'):
                                st.markdown("**📅 Key Dates:**")
                                for date in metadata['dates'][:5]:  # Show first 5 dates
                                    st.write(f"• {date}")
                        
                        st.markdown("---")
                        
                        # Gemini-generated summary with better presentation
                        if 'summary' in compliance_results:
                            st.markdown("### 🤖 Gemini AI Analysis Summary")
                            st.info(compliance_results['summary'])
                    
                    with tab2:
                        st.markdown("### 🚨 Compliance Issues")
                        
                        if 'issues' in compliance_results:
                            issues = compliance_results['issues']
                            
                            # Handle both dataclass and dict formats
                            if issues:
                                # Try to categorize by severity
                                high_issues = []
                                medium_issues = []
                                low_issues = []
                                
                                for issue in issues:
                                    if hasattr(issue, 'severity'):
                                        severity = issue.severity
                                    elif isinstance(issue, dict):
                                        severity = issue.get('severity', 'medium')
                                    else:
                                        severity = 'medium'
                                    
                                    if severity == 'high':
                                        high_issues.append(issue)
                                    elif severity == 'medium':
                                        medium_issues.append(issue)
                                    else:
                                        low_issues.append(issue)
                                
                                # Summary statistics
                                summary_cols = st.columns(3)
                                with summary_cols[0]:
                                    st.metric("🔴 High Priority", len(high_issues))
                                with summary_cols[1]:
                                    st.metric("🟡 Medium Priority", len(medium_issues))
                                with summary_cols[2]:
                                    st.metric("🟢 Low Priority", len(low_issues))
                                
                                st.markdown("---")
                                
                                # Display issues with improved organization
                                def display_issue(issue, icon):
                                    if hasattr(issue, 'rule_name'):
                                        title = issue.rule_name
                                        message = issue.violation_message
                                        elements = ', '.join(issue.missing_elements)
                                        confidence = issue.confidence_score
                                    elif isinstance(issue, dict):
                                        title = issue.get('rule_name', 'Unknown Issue')
                                        message = issue.get('violation_message', 'No description available')
                                        elements = issue.get('missing_elements', 'N/A')
                                        confidence = issue.get('confidence_score', 0.0)
                                    else:
                                        title = "Unknown Issue"
                                        message = str(issue)
                                        elements = "N/A"
                                        confidence = 0.0
                                    
                                    with st.expander(f"{icon} {title}"):
                                        st.markdown(f"**Issue:** {message}")
                                        st.markdown(f"**Missing Elements:** {elements}")
                                        st.markdown(f"**Confidence:** {confidence:.2f}")
                                
                                if high_issues:
                                    st.error(f"🔴 **High Priority Issues** ({len(high_issues)})")
                                    for issue in high_issues:
                                        display_issue(issue, "❗")
                                
                                if medium_issues:
                                    st.warning(f"🟡 **Medium Priority Issues** ({len(medium_issues)})")
                                    for issue in medium_issues:
                                        display_issue(issue, "⚠️")
                                
                                if low_issues:
                                    st.info(f"🟢 **Low Priority Issues** ({len(low_issues)})")
                                    for issue in low_issues:
                                        display_issue(issue, "ℹ️")
                            else:
                                st.success("✅ **No compliance issues detected!**")
                        else:
                            st.success("✅ **No compliance issues detected!**")
                    
                    with tab3:
                        st.markdown("### 🔍 Gemini Smart Search")
                        
                        # Enhanced search interface with Gemini intelligence
                        search_container = st.container()
                        with search_container:
                            st.markdown("#### 🧠 Intelligent Search powered by Gemini")
                            search_query = st.text_input(
                                "Search contract sections:",
                                placeholder="Ask natural language questions: 'What are the termination conditions?' or 'Show me payment terms'",
                                help="Use natural language to search through the contract with Gemini AI understanding"
                            )
                        
                        if search_query:
                            with st.spinner("🔍 Searching with Gemini intelligence..."):
                                try:
                                    # Use Gemini intelligent search
                                    gemini_search_results = agent.search_agent.intelligent_search(
                                        search_query, chunks
                                    )
                                    
                                    if gemini_search_results and not any('error' in str(result) for result in gemini_search_results):
                                        st.markdown(f"**🤖 Gemini found {len(gemini_search_results)} relevant sections:**")
                                        for i, result in enumerate(gemini_search_results, 1):
                                            section_title = result.get('section_title', f'Section {i}')
                                            relevance_score = result.get('relevance_score', 0)
                                            content = result.get('text', str(result))
                                            
                                            with st.expander(f"📄 {section_title} (Relevance: {relevance_score})"):
                                                st.text_area("Content", content, height=150, key=f"gemini_search_{i}")
                                    else:
                                        # Fallback to basic search
                                        st.info("🔄 Using basic search as fallback...")
                                        search_results = vector_store.search_similar(search_query, k=3)
                                        
                                        if search_results:
                                            st.markdown(f"**Found {len(search_results)} relevant sections:**")
                                            for i, result in enumerate(search_results, 1):
                                                with st.expander(f"📄 Result {i}"):
                                                    st.text_area("Content", result, height=150, key=f"basic_search_{i}")
                                        else:
                                            st.info("No relevant sections found for your search query.")
                                
                                except Exception as e:
                                    st.error(f"Search error: {str(e)}")
                                    st.info("Trying basic search as fallback...")
                                    search_results = vector_store.search_similar(search_query, k=3)
                                    
                                    if search_results:
                                        for i, result in enumerate(search_results, 1):
                                            with st.expander(f"📄 Result {i}"):
                                                st.text_area("Content", result, height=150, key=f"fallback_search_{i}")
                    
                    with tab4:
                        st.markdown("### 📊 Risk Assessment")
                        
                        if 'risk_assessment' in compliance_results:
                            risk_data = compliance_results['risk_assessment']
                            
                            # Risk score visualization with improved layout
                            st.markdown("#### 🎯 Overall Risk Metrics")
                            risk_metric_cols = st.columns(3)
                            with risk_metric_cols[0]:
                                st.metric(
                                    "Overall Risk Score",
                                    f"{risk_data.get('score', 0)}/100",
                                    delta=f"{risk_data.get('level', 'unknown').title()} Risk"
                                )
                            with risk_metric_cols[1]:
                                st.metric("Total Issues", risk_data.get('total_issues', 0))
                            with risk_metric_cols[2]:
                                st.metric("High Severity", risk_data.get('high_severity', 0))
                            
                            # Risk breakdown with better visualization
                            st.markdown("#### 📈 Risk Breakdown by Category")
                            risk_categories = risk_data.get('categories', {})
                            if risk_categories:
                                for category, score in risk_categories.items():
                                    st.progress(score / 100, text=f"{category.title()}: {score}%")
                            else:
                                st.info("No detailed risk breakdown available.")
                        else:
                            st.info("Risk assessment data not available.")
                    
                    st.markdown("---")
                    
                    # Interactive AI Chat with Gemini intelligence
                    st.markdown("## 🤖 Gemini AI Contract Assistant")
                    
                    # Chat interface with better styling
                    chat_container = st.container()
                    with chat_container:
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        
                        # Display chat history
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])
                        
                        # Chat input with Gemini intelligence
                        if prompt := st.chat_input("Ask Gemini about the contract..."):
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            
                            with st.chat_message("user"):
                                st.markdown(prompt)
                            
                            with st.chat_message("assistant"):
                                with st.spinner("🤖 Gemini AI analyzing..."):
                                    try:
                                        response = agent.analyze_contract_comprehensive(prompt, text)
                                        st.markdown(response)
                                        st.session_state.messages.append({"role": "assistant", "content": response})
                                    except Exception as e:
                                        error_msg = f"Sorry, I encountered an error while analyzing: {str(e)}"
                                        st.error(error_msg)
                                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    st.markdown("---")
                    
                    # Advanced Analysis Tools with Gemini enhancement
                    st.markdown("## 🔧 Advanced Gemini Analysis Tools")
                    
                    tools_cols = st.columns(3)
                    
                    with tools_cols[0]:
                        st.markdown("### 📊 Gemini Report")
                        if st.button("📊 Generate Gemini Report", use_container_width=True):
                            with st.spinner("Generating comprehensive Gemini report..."):
                                try:
                                    report = agent.generate_comprehensive_report(text)
                                    st.download_button(
                                        "📥 Download Gemini Report",
                                        report,
                                        file_name=f"gemini_contract_analysis_{uploaded_file.name}.md",
                                        mime="text/markdown",
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.error(f"Error generating report: {str(e)}")
                    
                    with tools_cols[1]:
                        st.markdown("### 🔍 Gemini Deep Analysis")
                        if st.button("🔍 Deep Gemini Analysis", use_container_width=True):
                            with st.spinner("Performing deep Gemini analysis..."):
                                try:
                                    deep_analysis = agent.perform_deep_analysis(text)
                                    st.json(deep_analysis)
                                except Exception as e:
                                    st.error(f"Error performing deep analysis: {str(e)}")
                    
                    # with tools_cols[2]:
                    #     st.markdown("### ⚖️ Gemini Risk Assessment")
                    #     if st.button("⚖️ Gemini Legal Risk Assessment", use_container_width=True):
                    #         with st.spinner("Assessing legal risks with Gemini..."):
                    #             try:
                    #                 legal_risks = agent.assess_legal_risks(text)
                    #                 st.markdown(legal_risks)
                    #             except Exception as e:
                    #                 st.error(f"Error assessing legal risks: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.exception(e)
    
    # Enhanced status column
    with status_col:
        st.markdown("## 📈 System Status")
        
        # Agent status with improved presentation
        with st.container():
            st.markdown("### 🤖 Agent Status")
            if 'agent' in locals():
                st.success("✅ **Enhanced Gemini Agent Active**")
                st.info(f"🔧 **Model:** {model_choice}")
                st.info(f"🔄 **Max Iterations:** {max_iterations}")
                st.info(f"📊 **Analysis Type:** {analysis_type}")
                st.info("🧠 **Gemini Intelligence:** Enabled")
            else:
                st.warning("⚠️ **Agent Not Initialized**")
        
        # Recent activity with better organization
        st.markdown("### 🕐 Recent Activity")
        with st.container():
            if 'uploaded_file' in locals() and uploaded_file:
                st.markdown(f"**📄 File:** {uploaded_file.name}")
                st.markdown(f"**📊 Analysis:** {analysis_type}")
                st.markdown(f"**✅ Status:** Complete")
                st.markdown("**🧠 Gemini:** Active")
            else:
                st.info("No recent activity")

if __name__ == "__main__":
    main()
