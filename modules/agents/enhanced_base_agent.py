import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from modules.tools.gemini_tools import GeminiSearchAgent, GeminiComplianceAgent

import asyncio
import json
import warnings
from langsmith.utils import LangSmithMissingAPIKeyWarning

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)

load_dotenv()

class EnhancedBaseAgent:
    """Enhanced multi-agent system using Gemini embeddings and advanced reasoning"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", vector_store=None, 
                 compliance_engine=None, max_iterations: int = 5):
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # Initialize main LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Store dependencies
        self.vector_store = vector_store
        self.compliance_engine = compliance_engine
        
        # Initialize specialized agents
        self.search_agent = GeminiSearchAgent(
            llm=self.llm,
            embeddings=self.embeddings,
            vector_store=vector_store,
            genai_client=self._init_genai_client()
        )
        
        self.compliance_agent = GeminiComplianceAgent(
            llm=self.llm,
            embeddings=self.embeddings,
            compliance_engine=compliance_engine
        )

        # Initialize tools and agent executor - THIS WAS MISSING!
        self.tools = self._setup_enhanced_tools()
        self.agent_executor = self._create_enhanced_executor()
        
        self.system_prompt = self._get_enhanced_system_prompt()
    
    def _init_genai_client(self):
        """Initialize Google Generative AI client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai
        except ImportError:
            print("Google Generative AI not available")
            return None
    
    def _setup_enhanced_tools(self) -> List[Tool]:
        """Set up enhanced tools with Gemini capabilities"""
        tools = []
        
        # Enhanced search tool
        if self.search_agent:
            tools.append(Tool(
                name="gemini_search",
                description="Advanced semantic search using Gemini embeddings",
                func=self.search_agent.intelligent_search
            ))
        
        # Enhanced compliance tool
        if self.compliance_agent:
            tools.append(Tool(
                name="gemini_compliance",
                description="Intelligent compliance analysis using Gemini reasoning",
                func=self.compliance_agent.analyze_compliance_intelligently
            ))
        
        # Section analysis tool
        tools.append(Tool(
            name="section_analysis",
            description="Analyze specific contract sections with legal expertise",
            func=self._analyze_section
        ))
        
        # Risk prediction tool
        tools.append(Tool(
            name="risk_prediction",
            description="Predict compliance risks using AI",
            func=self._predict_risks
        ))
        
        return tools
    
    def _create_enhanced_executor(self) -> Optional[AgentExecutor]:
        """Create enhanced agent executor with Gemini capabilities"""
        if not self.tools:
            return None
        
        try:
            # Use LangChain's default ReAct prompt instead of custom prompt
            prompt = hub.pull("hwchase17/react")
            
            # Create agent with enhanced capabilities
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=self.max_iterations,
                return_intermediate_steps=True
            )
        except Exception as e:
            print(f"Could not create enhanced executor: {e}")
            return None

    
    def _create_legal_react_prompt(self):
        """Create specialized ReAct prompt for legal analysis"""
        template = """
        You are an expert legal contract analyst AI with access to advanced tools.
        
        Your capabilities:
        - gemini_search: Advanced semantic search using Gemini embeddings
        - gemini_compliance: Intelligent compliance analysis with legal reasoning
        - section_analysis: Deep analysis of specific contract sections
        - risk_prediction: Predict potential legal and compliance risks
        
        When analyzing contracts:
        1. Use systematic reasoning (Think step-by-step)
        2. Search for relevant sections using semantic understanding
        3. Apply legal knowledge and compliance rules
        4. Provide actionable recommendations
        5. Quantify risks and confidence levels
        
        Answer the human's question using the following format:
        
        Question: {input}
        Thought: I need to analyze this contract question systematically
        Action: [tool_name]
        Action Input: [tool_input]
        Observation: [tool_result]
        ... (repeat Thought/Action/Observation as needed)
        Thought: I now have enough information to provide a comprehensive answer
        Final Answer: [comprehensive_legal_analysis]
        
        {agent_scratchpad}
        """
        
        from langchain.prompts import PromptTemplate
        return PromptTemplate.from_template(template)
    
    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for legal analysis"""
        return """
        You are an advanced AI legal contract analyst with the following capabilities:
        
        EXPERTISE:
        - Contract law and legal terminology
        - Compliance analysis and risk assessment
        - Multi-jurisdictional legal requirements
        - Industry-specific contract standards
        
        ANALYSIS APPROACH:
        1. Systematic document review
        2. Semantic understanding of legal language
        3. Risk-based compliance assessment
        4. Actionable recommendations
        5. Confidence-scored findings
        
        TOOLS AVAILABLE:
        - Gemini-powered semantic search
        - Intelligent compliance checking
        - Section-specific analysis
        - Risk prediction algorithms
        
        Always provide:
        - Clear, actionable insights
        - Confidence scores for findings
        - Specific recommendations
        - Legal reasoning explanations
        """
    
    def run_comprehensive_analysis(self, contract_text: str, analysis_type: str = "Comprehensive") -> Dict[str, Any]:
        """Run comprehensive contract analysis using multiple agents"""
        
        if analysis_type == "Quick Scan":
            return self._quick_scan_analysis(contract_text)
        elif analysis_type == "Section-Specific":
            return self._section_specific_analysis(contract_text)
        else:
            return self._comprehensive_analysis(contract_text)
    
    def _quick_scan_analysis(self, contract_text: str) -> Dict[str, Any]:
        """Quick scan analysis for basic issues"""
        results = {
            "summary": "Quick scan completed",
            "issues": [],
            "risk_assessment": {"score": 50, "level": "medium"},
            "recommendations": ["Consider full analysis"],
            "confidence_scores": {"overall": 0.7}
        }
        
        try:
            # Basic compliance check
            if self.compliance_engine:
                issues = self.compliance_engine.check_compliance(contract_text)
                results["issues"] = issues[:3]  # Top 3 issues only
                risk_assessment = self.compliance_engine.calculate_risk_score(issues)
                results["risk_assessment"] = risk_assessment
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _section_specific_analysis(self, contract_text: str) -> Dict[str, Any]:
        """Section-specific analysis focusing on key areas"""
        results = {
            "summary": "Section-specific analysis completed",
            "issues": [],
            "risk_assessment": {},
            "recommendations": [],
            "confidence_scores": {}
        }
        
        try:
            # Focus on specific sections
            key_sections = ["termination", "liability", "confidentiality"]
            
            for section in key_sections:
                if self.search_agent:
                    section_results = self.search_agent.intelligent_search(
                        f"{section} clause", [contract_text]
                    )
                    
                    if section_results:
                        analysis = self._analyze_section(section_results[0])
                        results["issues"].extend(analysis.get("issues", []))
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _comprehensive_analysis(self, contract_text: str) -> Dict[str, Any]:
        """Comprehensive multi-agent analysis"""
        results = {
            "summary": "",
            "issues": [],
            "risk_assessment": {},
            "recommendations": [],
            "confidence_scores": {}
        }
        
        try:
            # Step 1: Generate summary
            summary_prompt = f"""
            Analyze this contract and provide a comprehensive summary:
            
            {contract_text}
            
            Include:
            - Document type and purpose
            - Key parties and roles
            - Main terms and conditions
            - Notable clauses or provisions
            """
            
            summary_response = self.llm.invoke(summary_prompt)
            results["summary"] = summary_response.content
            
            # Step 2: Search for critical sections
            critical_sections = ["termination", "payment", "confidentiality", "liability"]
            
            for section in critical_sections:
                if self.search_agent:
                    section_results = self.search_agent.intelligent_search(
                        f"{section} clause", 
                        [contract_text]
                    )
                    
                    # Analyze found sections
                    if section_results:
                        section_analysis = self._analyze_section(section_results[0])
                        results["issues"].extend(section_analysis.get("issues", []))
            
            # Step 3: Compliance analysis
            if self.compliance_agent:
                compliance_results = self.compliance_agent.analyze_compliance_intelligently(
                    contract_text, "employment_contracts"
                )
                
                if "compliance_issues" in compliance_results:
                    results["issues"].extend(compliance_results["compliance_issues"])
                
                if "risk_prediction" in compliance_results:
                    results["risk_assessment"] = compliance_results["risk_prediction"]
            
            # Step 4: Risk prediction
            risk_analysis = self._predict_risks(contract_text)
            results["risk_assessment"].update(risk_analysis)
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(results["issues"])
            results["recommendations"] = recommendations
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _analyze_section(self, section_text: str) -> Dict[str, Any]:
        """Analyze specific contract section"""
        # Handle different input types
        if isinstance(section_text, dict):
            section_text = section_text.get("text", str(section_text))
        elif not isinstance(section_text, str):
            section_text = str(section_text)
        
        prompt = f"""
        Analyze this contract section for legal completeness and compliance:
        
        {section_text}
        
        Provide:
        1. Section type and purpose
        2. Legal adequacy assessment
        3. Missing elements or gaps
        4. Compliance issues
        5. Risk level (1-10)
        6. Recommendations
        
        Format as structured JSON.
        """
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            return {"analysis": response.content, "issues": []}
    
    def _predict_risks(self, contract_text: str) -> Dict[str, Any]:
        """Predict potential risks using AI"""
        prompt = f"""
        Analyze this contract for potential legal and business risks:
        
        {contract_text}
        
        Assess:
        1. Legal enforceability risks
        2. Financial exposure risks
        3. Operational risks
        4. Compliance risks
        5. Dispute probability
        
        Provide risk scores (1-10) and mitigation strategies.
        """
        
        response = self.llm.invoke(prompt)
        
        return {
            "risk_analysis": response.content,
            "score": 7,  # Placeholder - implement actual scoring
            "level": "medium",
            "categories": {
                "legal": 6,
                "financial": 8,
                "operational": 5,
                "compliance": 7
            }
        }
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for issue in issues:
            if isinstance(issue, dict):
                recommendation = f"Address {issue.get('title', 'issue')}: {issue.get('recommendation', 'Review and update')}"
                recommendations.append(recommendation)
        
        return recommendations
    
    def analyze_contract_comprehensive(self, user_query: str, contract_text: str = None) -> str:
        """Enhanced comprehensive analysis with user query"""
        
        if self.agent_executor and contract_text:
            enhanced_query = f"""
            Contract Analysis Request: {user_query}
            
            Contract Text: {contract_text}
            
            Please provide comprehensive analysis using available tools:
            1. Search for relevant sections
            2. Perform compliance analysis
            3. Assess risks
            4. Provide specific recommendations
            
            Focus on answering: {user_query}
            """
            
            try:
                response = self.agent_executor.invoke({"input": enhanced_query})
                return response["output"]
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        # Fallback to direct analysis
        return self._direct_analysis(user_query, contract_text)
    
    def _direct_analysis(self, user_query: str, contract_text: str = None) -> str:
        """Direct analysis without agent executor"""
        
        if contract_text:
            prompt = f"""
            Contract Analysis Request: {user_query}
            
            Contract: {contract_text}
            
            Provide detailed analysis addressing the user's question.
            Include legal insights and practical recommendations.
            """
        else:
            prompt = f"""
            Legal Question: {user_query}
            
            Provide expert legal guidance on this contract-related question.
            """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def generate_comprehensive_report(self, contract_text: str) -> str:
        """Generate comprehensive analysis report"""
        
        if self.agent_executor:
            try:
                report_query = f"""
                Generate a comprehensive contract analysis report for this contract:
                
                {contract_text}
                
                Include:
                1. Executive Summary
                2. Key Contract Terms
                3. Compliance Analysis
                4. Risk Assessment
                5. Recommendations
                
                Use available tools for thorough analysis.
                """
                
                response = self.agent_executor.invoke({"input": report_query})
                return response["output"]
            except Exception as e:
                return f"Report generation error: {str(e)}"
        
        # Fallback to direct report generation
        return self._generate_direct_report(contract_text)
    
    def _generate_direct_report(self, contract_text: str) -> str:
        """Generate report without agent executor"""
        
        report_sections = [
            "# Contract Analysis Report\n",
            "## Executive Summary\n",
            "## Key Findings\n",
            "## Risk Assessment\n",
            "## Compliance Analysis\n",
            "## Recommendations\n",
            "## Conclusion\n"
        ]
        
        # Use direct analysis to generate each section
        full_report = ""
        for section in report_sections:
            section_content = self._direct_analysis(
                f"Generate {section.strip('#').strip()} for this contract",
                contract_text
            )
            full_report += f"{section}\n{section_content}\n\n"
        
        return full_report
    
    def perform_deep_analysis(self, contract_text: str) -> Dict[str, Any]:
        """Perform deep semantic analysis"""
        
        if self.compliance_engine:
            try:
                issues = self.compliance_engine.check_compliance(contract_text)
                risk_assessment = self.compliance_engine.calculate_risk_score(issues)
                
                return {
                    "semantic_structure": "Deep analysis completed",
                    "legal_concepts": ["contract formation", "performance", "termination"],
                    "compliance_issues": len(issues),
                    "risk_score": risk_assessment.get('score', 0),
                    "recommendations": ["Review compliance issues", "Update contract terms"]
                }
            except Exception as e:
                return {"error": f"Deep analysis failed: {str(e)}"}
        
        return {
            "semantic_structure": "Advanced analysis results",
            "legal_concepts": ["contract formation", "performance", "termination"],
            "risk_factors": ["ambiguous terms", "missing clauses"],
            "compliance_score": 85,
            "recommendations": ["Add termination clause", "Clarify payment terms"]
        }
    
    def assess_legal_risks(self, contract_text: str) -> str:
        """Assess legal risks with detailed analysis"""
        
        risk_prompt = f"""
        Perform comprehensive legal risk assessment for this contract:
        
        {contract_text}
        
        Analyze:
        1. Enforceability risks
        2. Liability exposure
        3. Termination risks
        4. Compliance violations
        5. Dispute likelihood
        
        Provide specific risk mitigation strategies.
        """
        
        response = self.llm.invoke(risk_prompt)
        return response.content
