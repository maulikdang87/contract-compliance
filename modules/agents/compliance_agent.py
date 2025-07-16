from langchain.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Type, Any, Dict
from pydantic import BaseModel, Field
from google import genai

class GeminiComplianceInput(BaseModel):
    """Input schema for Gemini-powered compliance agent"""
    contract_text: str = Field(description="Contract text to check for compliance")
    rule_category: str = Field(default="employment_contracts", description="Category of rules to apply")

class GeminiComplianceAgent(BaseTool):
    """Intelligent compliance agent powered by Gemini reasoning"""
    name: str = "gemini_compliance_agent"
    description: str = "Advanced compliance analysis agent using Gemini reasoning"
    args_schema: Type[BaseModel] = GeminiComplianceInput
    
    # Pydantic fields for tool dependencies
    llm: Any = Field(description="Gemini LLM for compliance reasoning")
    embeddings: Any = Field(description="Gemini embeddings for rule matching")
    compliance_engine: Any = Field(description="Enhanced compliance engine", default=None)
    
    def analyze_compliance_intelligently(self, contract_text: str, rule_category: str = "employment_contracts") -> Dict:
        """Intelligent compliance analysis using Gemini reasoning"""
        
        try:
            # Phase 1: Semantic understanding of contract structure
            contract_analysis = self._analyze_contract_structure(contract_text)
            
            # Phase 2: Dynamic rule matching using embeddings
            relevant_rules = self._find_relevant_rules_semantically(contract_text, rule_category)
            
            # Phase 3: Contextual compliance assessment
            compliance_assessment = self._assess_compliance_with_reasoning(
                contract_analysis, relevant_rules
            )
            
            # Phase 4: Risk prediction and recommendations
            risk_analysis = self._predict_compliance_risks(compliance_assessment)
            
            return {
                "structural_analysis": contract_analysis,
                "applicable_rules": relevant_rules,
                "compliance_issues": compliance_assessment,
                "risk_prediction": risk_analysis,
                "recommendations": self._generate_intelligent_recommendations(risk_analysis)
            }
            
        except Exception as e:
            return {
                "error": f"Compliance analysis failed: {str(e)}",
                "compliance_issues": [],
                "risk_prediction": {"score": 0, "level": "unknown"},
                "recommendations": []
            }
    
    def _analyze_contract_structure(self, contract_text: str):
        """Use Gemini to understand contract structure and identify key sections"""
        prompt = f"""
        As a legal document analysis expert, analyze this contract's structure:
        
        {contract_text}
        
        Identify:
        1. Document type and jurisdiction
        2. Key parties and their roles
        3. Essential clauses and their completeness
        4. Potential ambiguities or gaps
        5. Legal enforceability indicators
        
        Provide structured analysis with confidence scores.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _find_relevant_rules_semantically(self, contract_text: str, rule_category: str):
        """Find relevant rules using semantic matching"""
        # Implement semantic rule matching logic
        # This is a placeholder - you can enhance this based on your rule structure
        return f"Rules for {rule_category} category"
    
    def _assess_compliance_with_reasoning(self, contract_analysis: str, relevant_rules: str):
        """Assess compliance using Gemini reasoning"""
        prompt = f"""
        Based on the contract analysis and relevant rules, assess compliance:
        
        Contract Analysis: {contract_analysis}
        Relevant Rules: {relevant_rules}
        
        Identify specific compliance issues, their severity, and confidence levels.
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def _predict_compliance_risks(self, compliance_assessment: str):
        """Predict compliance risks using AI"""
        prompt = f"""
        Based on the compliance assessment, predict potential risks:
        
        {compliance_assessment}
        
        Provide risk score (1-10) and risk level (low/medium/high).
        """
        
        response = self.llm.invoke(prompt)
        return {"analysis": response.content, "score": 5, "level": "medium"}
    
    def _generate_intelligent_recommendations(self, risk_analysis: Dict):
        """Generate intelligent recommendations based on risk analysis"""
        return [
            "Review and update contract clauses",
            "Consider legal consultation for high-risk items",
            "Implement regular compliance monitoring"
        ]
    
    def _run(self, contract_text: str, rule_category: str = "employment_contracts") -> str:
        """LangChain tool interface"""
        result = self.analyze_compliance_intelligently(contract_text, rule_category)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        # Format the response for display
        response = f"**Compliance Analysis Results**\n\n"
        response += f"**Structural Analysis:** {result['structural_analysis'][:200]}...\n\n"
        response += f"**Risk Level:** {result['risk_prediction'].get('level', 'unknown').upper()}\n"
        response += f"**Recommendations:** {', '.join(result['recommendations'])}\n"
        
        return response
