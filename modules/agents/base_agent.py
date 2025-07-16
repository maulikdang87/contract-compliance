import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class BaseContractAgent:
    """
    Enhanced contract agent with tool integration and structured analysis.
    
    This version combines your structured prompting approach with the
    tool integration capabilities we built in Week 3.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash", vector_store=None, compliance_engine=None):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # Store tool dependencies
        self.vector_store = vector_store
        self.compliance_engine = compliance_engine
        
        # Initialize tools if dependencies are provided
        self.tools = self._setup_tools() if vector_store and compliance_engine else []
        
        # Create agent executor if tools are available
        self.agent_executor = self._create_agent_executor() if self.tools else None
        
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        return """You are a contract compliance expert AI assistant. Your role is to:
        1. Analyze contract documents for compliance issues
        2. Identify potential legal risks with specific scores
        3. Provide clear, actionable recommendations
        4. Answer in structured format when requested
        
        Always be thorough, accurate, and provide specific examples when identifying issues.
        When asked for structured analysis, fill in the blanks with precise, concise answers."""
    
    def _setup_tools(self) -> List[Tool]:
        """Set up tools for agent if dependencies are available"""
        if not self.vector_store or not self.compliance_engine:
            return []
        
        try:
            from modules.tools.contract_tools import ContractSearchTool, ComplianceCheckTool
            
            return [
                ContractSearchTool(vector_store=self.vector_store),  # ✅ Keyword argument
                ComplianceCheckTool(compliance_engine=self.compliance_engine)  # ✅ Keyword argument
            ]
        except ImportError as e:
            print(f"Could not import tools: {e}")
            return []

    
    def _create_agent_executor(self) -> Optional[AgentExecutor]:
        """Create ReAct agent executor with tools"""
        if not self.tools:
            return None
        
        try:
            # Get the ReAct prompt from LangChain hub
            prompt = hub.pull("hwchase17/react")
            
            # Create the agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            # Create executor
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            print(f"Could not create agent executor: {e}")
            return None
    
    def analyze_contract(self, contract_text: str, specific_focus: str = None) -> str:
        """
        Main contract analysis method - uses tools if available, otherwise direct analysis
        """
        
        # If we have tools available, use the enhanced agent
        if self.agent_executor:
            return self._analyze_with_tools(contract_text, specific_focus)
        
        # Otherwise, use direct analysis
        return self._analyze_direct(contract_text, specific_focus)
    
    def analyze_contract_comprehensive(self, user_query: str, contract_text: str = None) -> str:
        """
        FIXED: Comprehensive analysis method that matches main.py expectations
        This method handles both user queries and contract analysis
        """
        
        # If we have tools available and contract text, use enhanced analysis
        if self.agent_executor and contract_text:
            enhanced_query = f"""
            User Question: {user_query}
            
            Contract Text: {contract_text}
            
            Please analyze the contract to answer the user's question comprehensively.
            Use available tools to search for relevant sections and check compliance.
            
            Provide a detailed response that directly addresses the user's question.
            """
            
            try:
                response = self.agent_executor.invoke({"input": enhanced_query})
                return response["output"]
            except Exception as e:
                # Fallback to direct analysis
                return self._analyze_direct_query(user_query, contract_text)
        
        # If no contract text provided, treat as general query
        elif self.agent_executor:
            try:
                response = self.agent_executor.invoke({"input": user_query})
                return response["output"]
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}"
        
        # Fallback to direct analysis
        else:
            return self._analyze_direct_query(user_query, contract_text)
    
    def _analyze_direct_query(self, user_query: str, contract_text: str = None) -> str:
        """Handle direct query analysis without tools"""
        
        if contract_text:
            content = f"""
            Please analyze this contract to answer the user's question:
            
            User Question: {user_query}
            
            Contract Text: {contract_text}
            
            Provide a detailed response that directly addresses the question.
            """
        else:
            content = f"""
            Please provide information about: {user_query}
            
            Focus on contract compliance and legal considerations.
            """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _analyze_with_tools(self, contract_text: str, specific_focus: str = None) -> str:
        """Enhanced analysis using ReAct agent with tools"""
        
        focus_instruction = f"\nPay special attention to: {specific_focus}" if specific_focus else ""
        
        query = f"""
        Analyze this contract comprehensively using available tools:
        
        {contract_text}
        {focus_instruction}
        
        Please provide:
        1. Contract Summary
        2. Compliance Issues (use compliance_check tool)
        3. Risk Assessment with scores
        4. Specific Recommendations
        
        Use the contract_search tool to find relevant sections as needed.
        """
        
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            # Fallback to direct analysis if tools fail
            return self._analyze_direct(contract_text, specific_focus)
    
    def _analyze_direct(self, contract_text: str, specific_focus: str = None) -> str:
        """Direct analysis without tools (fallback method)"""
        
        focus_instruction = f"\nPay special attention to: {specific_focus}" if specific_focus else ""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Please analyze the following contract text for compliance issues:
            
            {contract_text}
            {focus_instruction}
            
            Provide your analysis in the following format:
            1. Contract Summary
            2. Identified Issues
            3. Risk Assessment
            4. Recommendations
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def analyze_structured(self, contract_text: str) -> str:
        """
        Structured analysis with specific fill-in-the-blank format
        """
        
        structured_prompt = f"""
        You are a contract analysis agent. Answer these specific questions about the contract:

        RISK ASSESSMENT:
        1. Contract enforceability score (1-10): ___
        2. Highest financial risk exposure: ___
        3. Most likely dispute scenario: ___

        MISSING CLAUSES:
        4. List exactly 3 critical missing clauses:
           - ___
           - ___
           - ___

        IMMEDIATE ACTIONS:
        5. Most urgent fix needed: ___
        6. Estimated cost to remedy top issues: ___
        7. Days to implement fixes: ___

        Format: Fill in blanks with concise answers only.
        
        CONTRACT TEXT:
        {contract_text}
        """
        
        response = self.llm.invoke(structured_prompt)
        return response.content
    
    def analyze_risk_score(self, contract_text: str) -> str:
        """Get specific risk score analysis"""
        
        prompt = f"""Rate this contract's enforceability (1-10 scale):
        10 = Fully enforceable, comprehensive
        1 = Unenforceable, major gaps
        
        Score: ___
        Primary weakness: ___
        Secondary concerns: ___
        
        Contract: {contract_text}"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def identify_critical_gaps(self, contract_text: str) -> str:
        """Identify missing clauses"""
        
        prompt = f"""List exactly 3 most critical missing clauses:
        1. ___
        2. ___
        3. ___
        
        Each answer must be 3-5 words maximum.
        
        Contract: {contract_text}"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_action_plan(self, contract_text: str) -> str:
        """Get immediate action recommendations"""
        
        prompt = f"""Provide immediate next steps:
        Most urgent fix: ___
        Estimated cost: ___
        Timeline: ___ days
        Priority level: ___
        
        Keep answers brief and specific.
        
        Contract: {contract_text}"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def extract_key_clauses(self, contract_text: str) -> Dict[str, str]:
        """Extract key clauses from contract"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Extract the key clauses from this contract and categorize them:
            
            {contract_text}
            
            Return the clauses in JSON format with categories like:
            - termination_clauses
            - payment_terms
            - liability_clauses
            - confidentiality_clauses
            - governing_law
            """)
        ]
        
        response = self.llm.invoke(messages)
        return {"extracted_clauses": response.content}
    
    def comprehensive_analysis(self, contract_text: str) -> Dict[str, str]:
        """
        Run all analysis methods and return comprehensive results
        """
        
        results = {
            "general_analysis": self.analyze_contract(contract_text),
            "structured_analysis": self.analyze_structured(contract_text),
            "risk_score": self.analyze_risk_score(contract_text),
            "critical_gaps": self.identify_critical_gaps(contract_text),
            "action_plan": self.get_action_plan(contract_text),
            "key_clauses": self.extract_key_clauses(contract_text)
        }
        
        return results
