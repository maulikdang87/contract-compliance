from langchain.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field

class ContractSearchInput(BaseModel):
    """Input schema for contract search tool"""
    query: str = Field(description="Search query to find relevant contract sections")
    max_results: int = Field(default=3, description="Maximum number of results to return")

class ContractSearchTool(BaseTool):
    """Tool for semantic search within contract documents"""
    name: str = "contract_search"
    description: str = "Search for specific clauses or sections within the uploaded contract"
    args_schema: Type[BaseModel] = ContractSearchInput
    vector_store: Any  # ✅ Declare vector_store as a field with type annotation

    def _run(self, query: str, max_results: int = 3) -> str:
        """Execute the search and return formatted results"""
        try:
            results = self.vector_store.search_similar(query, k=max_results)
            
            if not results:
                return f"No relevant sections found for query: {query}"
            
            formatted_results = []
            for i, chunk in enumerate(results, 1):
                formatted_results.append(f"Section {i}:\n{chunk}\n")
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"Error searching contract: {str(e)}"

class ComplianceCheckInput(BaseModel):
    """Input schema for compliance checking tool"""
    contract_text: str = Field(description="Contract text to check for compliance")
    rule_category: str = Field(default="employment_contracts", description="Category of rules to apply")

class ComplianceCheckTool(BaseTool):
    """Tool for systematic compliance checking"""
    name: str = "compliance_check"
    description: str = "Check contract text against compliance rules and regulations"
    args_schema: Type[BaseModel] = ComplianceCheckInput
    compliance_engine: Any  # ✅ Declare compliance_engine as a field with type annotation

    def _run(self, contract_text: str, rule_category: str = "employment_contracts") -> str:
        """Execute compliance check and return formatted results"""
        try:
            issues = self.compliance_engine.check_compliance(contract_text, rule_category)
            risk_assessment = self.compliance_engine.calculate_risk_score(issues)
            
            if not issues:
                return "✅ No compliance issues found. Contract appears to meet basic requirements."
            
            # Format the results for the AI agent
            result = f"🚨 Found {len(issues)} compliance issues:\n\n"
            result += f"Overall Risk Score: {risk_assessment['score']}/100 ({risk_assessment['level']} risk)\n\n"
            
            for issue in issues:
                result += f"• **{issue.rule_name}** ({issue.severity} severity)\n"
                result += f"  Issue: {issue.violation_message}\n"
                result += f"  Missing: {', '.join(issue.missing_elements)}\n"
                result += f"  Confidence: {issue.confidence_score:.2f}\n\n"
            
            return result
        
        except Exception as e:
            return f"Error checking compliance: {str(e)}"
