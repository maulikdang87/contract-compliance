from langchain_google_genai import ChatGoogleGenerativeAI
from modules.agents.embeddings_agent import GeminiEmbeddingSearchAgent
from modules.agents.compliance_agent import GeminiComplianceAgent

class GeminiAgentOrchestrator:
    def __init__(self, api_key: str):
        self.search_agent = GeminiEmbeddingSearchAgent(api_key)
        self.compliance_agent = GeminiComplianceAgent(api_key)
        self.coordination_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key
        )
        
    def orchestrate_contract_analysis(self, contract_text: str, user_query: str):
        """Coordinate multiple Gemini agents for comprehensive analysis"""
        
        # Agent 1: Semantic Search Agent
        relevant_sections = self.search_agent.intelligent_search(
            query=user_query,
            contract_sections=self._chunk_contract(contract_text)
        )
        
        # Agent 2: Compliance Agent
        compliance_analysis = self.compliance_agent.analyze_compliance_intelligently(
            contract_text=contract_text,
            rule_category="employment_contracts"
        )
        
        # Agent 3: Coordination Agent (decides next steps)
        coordination_decision = self._coordinate_agents(
            user_query, relevant_sections, compliance_analysis
        )
        
        return self._synthesize_final_response(coordination_decision)
