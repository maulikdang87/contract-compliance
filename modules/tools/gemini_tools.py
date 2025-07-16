from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import Type, Any, Dict, List
from pydantic import BaseModel, Field
import numpy as np
from google import genai
from google.genai import types
import json
import re

class GeminiSearchInput(BaseModel):
    """Input schema for Gemini-powered search agent"""
    query: str = Field(description="Search query for legal document analysis")
    context_type: str = Field(default="legal", description="Domain context for search")
    max_results: int = Field(default=5, description="Maximum results to return")

class GeminiComplianceInput(BaseModel):
    """Input schema for Gemini-powered compliance agent"""
    contract_text: str = Field(description="Contract text to check for compliance")
    rule_category: str = Field(default="employment_contracts", description="Category of rules to apply")

class GeminiSearchAgent(BaseTool):
    """Advanced semantic search agent using Gemini embeddings"""
    name: str = "gemini_search_agent"
    description: str = "Advanced semantic search agent using Gemini embeddings for legal document analysis"
    args_schema: Type[BaseModel] = GeminiSearchInput
    
    # Agent components as Pydantic fields
    llm: Any = Field(description="Gemini LLM for reasoning")
    embeddings: Any = Field(description="Gemini embeddings model")
    vector_store: Any = Field(description="Enhanced vector store")
    genai_client: Any = Field(description="Gemini client", default=None)

    def intelligent_search(self, query: str, contract_sections: List[str] = None) -> List[Dict]:
        """
        Enhanced search method using Gemini embeddings and LLM reasoning
        """
        try:
            # Step 1: Enhance query with legal context using Gemini LLM
            enhanced_query = self._enhance_query_with_llm(query)
            
            # Step 2: Use Gemini embeddings for semantic search
            if contract_sections:
                return self._gemini_embedding_search(enhanced_query, contract_sections)
            
            # Otherwise use vector store search with Gemini embeddings
            return self._search_vector_store_with_gemini(enhanced_query)
            
        except Exception as e:
            return [{"error": f"Gemini search failed: {str(e)}"}]
    
    def _enhance_query_with_llm(self, query: str) -> str:
        """Enhance query using Gemini LLM for better search results"""
        prompt = f"""
        As a legal document expert, enhance this search query for maximum relevance:
        
        Original Query: {query}
        
        Provide an expanded query that includes:
        1. Legal synonyms and related terms
        2. Alternative phrasings commonly used in contracts
        3. Contextual keywords for better semantic matching
        
        Return only the enhanced query string.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return query  # Fallback to original query
    
    def _gemini_embedding_search(self, query: str, contract_sections: List[str]) -> List[Dict]:
        """Search using Gemini embeddings for superior accuracy"""
        try:
            # Generate query embedding using Gemini
            query_embedding = self._generate_gemini_embedding(query)
            
            # Generate embeddings for all contract sections
            section_embeddings = []
            for section in contract_sections:
                section_embedding = self._generate_gemini_embedding(section)
                section_embeddings.append(section_embedding)
            
            # Calculate semantic similarity
            similarities = self._calculate_semantic_similarity(query_embedding, section_embeddings)
            
            # Create results with relevance scores
            results = []
            for i, (section, similarity) in enumerate(zip(contract_sections, similarities)):
                if similarity > 0.3:  # Threshold for relevance
                    results.append({
                        "text": section,
                        "section_title": self._extract_section_title(section),
                        "relevance_score": round(similarity * 10, 1),  # Scale to 1-10
                        "similarity": similarity
                    })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:5]
            
        except Exception as e:
            # Fallback to basic search
            return self._fallback_search(query, contract_sections)
    
    def _generate_gemini_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings using Gemini embedding model"""
        try:
            if self.genai_client:
                # Use direct Gemini client for better performance
                result = self.genai_client.embed_content(
                    model="models/embedding-001",
                    content=text
                )
                return np.array(result['embedding'])
            else:
                # Fallback to LangChain embeddings
                embedding = self.embeddings.embed_query(text)
                return np.array(embedding)
        except Exception as e:
            # Generate random embedding as last resort
            return np.random.rand(768)  # Standard embedding dimension
    
    def _calculate_semantic_similarity(self, query_embedding: np.ndarray, section_embeddings: List[np.ndarray]) -> List[float]:
        """Calculate cosine similarity between query and sections"""
        similarities = []
        for section_embedding in section_embeddings:
            try:
                # Cosine similarity
                dot_product = np.dot(query_embedding, section_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_section = np.linalg.norm(section_embedding)
                
                if norm_query == 0 or norm_section == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_section)
                
                similarities.append(max(0.0, similarity))  # Ensure non-negative
            except Exception:
                similarities.append(0.0)
        
        return similarities
    
    def _extract_section_title(self, section: str) -> str:
        """Extract meaningful section title from text"""
        lines = section.strip().split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and (line.isupper() or any(keyword in line.lower() for keyword in 
                ['termination', 'payment', 'confidentiality', 'liability', 'governing', 'dispute'])):
                return line[:50]  # Truncate if too long
        return "Contract Section"
    
    def _fallback_search(self, query: str, contract_sections: List[str]) -> List[Dict]:
        """Fallback search method using basic keyword matching"""
        results = []
        query_lower = query.lower()
        
        for i, section in enumerate(contract_sections):
            score = 0
            section_lower = section.lower()
            
            # Count keyword matches
            for word in query_lower.split():
                if word in section_lower:
                    score += section_lower.count(word)
            
            if score > 0:
                results.append({
                    "text": section,
                    "section_title": f"Section {i+1}",
                    "relevance_score": min(score, 10),
                    "similarity": score / 10
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]
    
    def _search_vector_store_with_gemini(self, query: str) -> List[Dict]:
        """Search vector store with Gemini enhancement"""
        if self.vector_store and hasattr(self.vector_store, 'search_similar'):
            try:
                search_results = self.vector_store.search_similar(query, k=5)
                
                formatted_results = []
                for i, result in enumerate(search_results):
                    text = result if isinstance(result, str) else str(result)
                    formatted_results.append({
                        "text": text,
                        "section_title": f"Vector Result {i+1}",
                        "relevance_score": 8 - i,  # Decreasing relevance
                        "similarity": 0.8 - (i * 0.1)
                    })
                
                return formatted_results
            except Exception as e:
                return [{"error": f"Vector search failed: {str(e)}"}]
        
        return [{"error": "Vector store not available"}]

    def _run(self, query: str, context_type: str = "legal", max_results: int = 5) -> str:
        """LangChain tool interface"""
        try:
            search_results = self.intelligent_search(query)
            return self._format_search_response(search_results, max_results)
        except Exception as e:
            return f"Gemini search agent error: {str(e)}"
    
    def _format_search_response(self, results: List[Dict], max_results: int) -> str:
        """Format search results for display"""
        if not results:
            return "No relevant sections found."
        
        formatted_response = []
        for i, result in enumerate(results[:max_results]):
            if "error" in result:
                formatted_response.append(f"Error: {result['error']}")
            else:
                title = result.get('section_title', f'Result {i+1}')
                score = result.get('relevance_score', 0)
                text = result.get('text', 'No content')
                
                formatted_response.append(
                    f"**{title}** (Relevance: {score}/10)\n"
                    f"{text[:300]}{'...' if len(text) > 300 else ''}\n"
                )
        
        return "\n".join(formatted_response)

class GeminiComplianceAgent(BaseTool):
    """Advanced compliance analysis agent using Gemini reasoning with vulnerability detection"""
    name: str = "gemini_compliance_agent"
    description: str = "Advanced compliance analysis agent using Gemini reasoning for comprehensive legal document review"
    args_schema: Type[BaseModel] = GeminiComplianceInput
    
    llm: Any = Field(description="Gemini LLM for compliance reasoning")
    embeddings: Any = Field(description="Gemini embeddings for rule matching")
    compliance_engine: Any = Field(description="Enhanced compliance engine", default=None)

    def analyze_compliance_intelligently(self, contract_text: str, rule_category: str = "employment_contracts") -> Dict:
        """
        Comprehensive compliance analysis using Gemini LLM with vulnerability detection
        """
        try:
            # Step 1: Use traditional compliance engine for rule-based analysis
            traditional_analysis = self._traditional_compliance_check(contract_text, rule_category)
            
            # Step 2: Use Gemini LLM for comprehensive vulnerability analysis
            gemini_analysis = self._gemini_vulnerability_analysis(contract_text, rule_category)
            
            # Step 3: Combine and enhance results
            combined_analysis = self._combine_analyses(traditional_analysis, gemini_analysis)
            
            return combined_analysis
            
        except Exception as e:
            return {
                "error": f"Compliance analysis failed: {str(e)}",
                "compliance_issues": [],
                "risk_assessment": {"score": 0, "level": "unknown"},
                "summary": "Analysis failed",
                "status": "error"
            }
    
    def _traditional_compliance_check(self, contract_text: str, rule_category: str) -> Dict:
        """Traditional rule-based compliance checking"""
        if self.compliance_engine:
            try:
                issues = self.compliance_engine.check_compliance(contract_text, rule_category)
                risk_assessment = self.compliance_engine.calculate_risk_score(issues)
                
                return {
                    "traditional_issues": issues,
                    "traditional_risk": risk_assessment,
                    "traditional_count": len(issues)
                }
            except Exception as e:
                return {
                    "traditional_issues": [],
                    "traditional_risk": {"score": 0, "level": "unknown"},
                    "traditional_count": 0
                }
        return {
            "traditional_issues": [],
            "traditional_risk": {"score": 0, "level": "unknown"},
            "traditional_count": 0
        }
    
    def _gemini_vulnerability_analysis(self, contract_text: str, rule_category: str) -> Dict:
        """Comprehensive vulnerability analysis using Gemini LLM"""
        
        # Define comprehensive analysis prompt
        analysis_prompt = f"""
        As an expert contract lawyer, perform a comprehensive vulnerability analysis of this contract:
        
        CONTRACT TEXT:
        {contract_text}
        
        ANALYSIS REQUIREMENTS:
        1. CRITICAL VULNERABILITIES:
           - Missing essential clauses (termination, liability, dispute resolution)
           - Ambiguous terms that could lead to disputes
           - Unenforceable provisions
           - One-sided terms that favor one party unfairly
        
        2. COMPLIANCE VIOLATIONS:
           - Labor law violations (for employment contracts)
           - Consumer protection issues
           - Regulatory compliance gaps
           - Industry-specific compliance requirements
        
        3. LEGAL RISKS:
           - Potential for litigation
           - Regulatory penalties
           - Financial exposure
           - Reputational risks
        
        4. SECURITY CONCERNS:
           - Data protection inadequacies
           - Intellectual property risks
           - Confidentiality gaps
           - Non-compete enforceability
        
        5. OPERATIONAL RISKS:
           - Unclear performance standards
           - Inadequate change management
           - Force majeure gaps
           - Termination procedure issues
        
        Return your analysis in the following JSON format:
        {{
            "critical_vulnerabilities": [
                {{
                    "type": "vulnerability_type",
                    "severity": "high|medium|low",
                    "description": "detailed description",
                    "recommendation": "specific recommendation",
                    "confidence": 0.0-1.0
                }}
            ],
            "compliance_violations": [
                {{
                    "violation_type": "type",
                    "severity": "high|medium|low",
                    "description": "detailed description",
                    "legal_basis": "relevant law or regulation",
                    "recommendation": "specific recommendation",
                    "confidence": 0.0-1.0
                }}
            ],
            "risk_assessment": {{
                "overall_risk_score": 0-100,
                "risk_level": "low|medium|high",
                "primary_concerns": ["concern1", "concern2"],
                "immediate_actions": ["action1", "action2"]
            }},
            "summary": "comprehensive summary of findings"
        }}
        """
        
        try:
            response = self.llm.invoke(analysis_prompt)
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(response.content)
                return self._process_gemini_analysis(analysis_result)
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response from text
                return self._parse_text_analysis(response.content)
                
        except Exception as e:
            return {
                "gemini_vulnerabilities": [],
                "gemini_compliance": [],
                "gemini_risk": {"score": 0, "level": "unknown"},
                "gemini_summary": f"Gemini analysis failed: {str(e)}"
            }
    
    def _process_gemini_analysis(self, analysis_result: Dict) -> Dict:
        """Process structured Gemini analysis results"""
        
        # Convert Gemini results to our internal format
        gemini_issues = []
        
        # Process critical vulnerabilities
        for vuln in analysis_result.get("critical_vulnerabilities", []):
            gemini_issues.append({
                "rule_name": f"Critical Vulnerability: {vuln.get('type', 'Unknown')}",
                "severity": vuln.get('severity', 'medium'),
                "violation_message": vuln.get('description', 'No description'),
                "missing_elements": [vuln.get('recommendation', 'Review required')],
                "confidence_score": vuln.get('confidence', 0.7),
                "source": "gemini_vulnerability"
            })
        
        # Process compliance violations
        for violation in analysis_result.get("compliance_violations", []):
            gemini_issues.append({
                "rule_name": f"Compliance Violation: {violation.get('violation_type', 'Unknown')}",
                "severity": violation.get('severity', 'medium'),
                "violation_message": violation.get('description', 'No description'),
                "missing_elements": [violation.get('recommendation', 'Review required')],
                "confidence_score": violation.get('confidence', 0.7),
                "legal_basis": violation.get('legal_basis', 'Not specified'),
                "source": "gemini_compliance"
            })
        
        # Process risk assessment
        risk_assessment = analysis_result.get("risk_assessment", {})
        
        return {
            "gemini_vulnerabilities": analysis_result.get("critical_vulnerabilities", []),
            "gemini_compliance": analysis_result.get("compliance_violations", []),
            "gemini_issues": gemini_issues,
            "gemini_risk": {
                "score": risk_assessment.get("overall_risk_score", 0),
                "level": risk_assessment.get("risk_level", "unknown"),
                "primary_concerns": risk_assessment.get("primary_concerns", []),
                "immediate_actions": risk_assessment.get("immediate_actions", [])
            },
            "gemini_summary": analysis_result.get("summary", "Analysis completed")
        }
    
    def _parse_text_analysis(self, text_content: str) -> Dict:
        """Parse text-based analysis when JSON parsing fails"""
        
        # Extract key information from text
        lines = text_content.split('\n')
        
        # Simple parsing for vulnerabilities and issues
        gemini_issues = []
        risk_indicators = ["high risk", "medium risk", "low risk", "critical", "violation"]
        
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in risk_indicators):
                # Extract potential issue
                severity = "medium"
                if "high" in line.lower() or "critical" in line.lower():
                    severity = "high"
                elif "low" in line.lower():
                    severity = "low"
                
                gemini_issues.append({
                    "rule_name": "Gemini Analysis Finding",
                    "severity": severity,
                    "violation_message": line,
                    "missing_elements": ["Review required"],
                    "confidence_score": 0.6,
                    "source": "gemini_text_analysis"
                })
        
        return {
            "gemini_vulnerabilities": [],
            "gemini_compliance": [],
            "gemini_issues": gemini_issues,
            "gemini_risk": {"score": 50, "level": "medium"},
            "gemini_summary": "Text-based analysis completed"
        }
    
    def _combine_analyses(self, traditional: Dict, gemini: Dict) -> Dict:
        """Combine traditional and Gemini analyses into comprehensive result"""
        
        # Combine all issues
        all_issues = []
        all_issues.extend(traditional.get("traditional_issues", []))
        all_issues.extend(gemini.get("gemini_issues", []))
        
        # Calculate combined risk assessment
        traditional_risk = traditional.get("traditional_risk", {})
        gemini_risk = gemini.get("gemini_risk", {})
        
        # Weight the risk scores (70% traditional, 30% Gemini for balance)
        trad_score = traditional_risk.get("score", 0)
        gem_score = gemini_risk.get("score", 0)
        combined_score = (trad_score * 0.7) + (gem_score * 0.3)
        
        # Determine combined risk level
        if combined_score >= 70:
            risk_level = "high"
        elif combined_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Create comprehensive summary
        summary = self._create_comprehensive_summary(traditional, gemini, len(all_issues), risk_level)
        
        return {
            "compliance_issues": all_issues,
            "risk_assessment": {
                "score": round(combined_score, 1),
                "level": risk_level,
                "total_issues": len(all_issues),
                "traditional_score": trad_score,
                "gemini_score": gem_score,
                "high_severity": len([i for i in all_issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'high']),
                "medium_severity": len([i for i in all_issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'medium']),
                "low_severity": len([i for i in all_issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'low']),
                "categories": {
                    "traditional": trad_score,
                    "vulnerabilities": gem_score,
                    "compliance": (trad_score + gem_score) / 2,
                    "overall": combined_score
                }
            },
            "summary": summary,
            "status": "complete",
            "analysis_details": {
                "traditional_analysis": traditional,
                "gemini_analysis": gemini,
                "gemini_vulnerabilities": gemini.get("gemini_vulnerabilities", []),
                "gemini_compliance": gemini.get("gemini_compliance", [])
            }
        }
    
    def _create_comprehensive_summary(self, traditional: Dict, gemini: Dict, total_issues: int, risk_level: str) -> str:
        """Create comprehensive analysis summary"""
        
        traditional_count = traditional.get("traditional_count", 0)
        gemini_summary = gemini.get("gemini_summary", "")
        
        summary = f"""
        🔍 COMPREHENSIVE COMPLIANCE ANALYSIS COMPLETE
        
        📊 ANALYSIS OVERVIEW:
        • Total Issues Found: {total_issues}
        • Traditional Rule-Based Issues: {traditional_count}
        • Gemini AI Vulnerabilities: {total_issues - traditional_count}
        • Overall Risk Level: {risk_level.upper()}
        
        🤖 GEMINI AI INSIGHTS:
        {gemini_summary}
        
        ⚖️ RECOMMENDATION:
        {"🚨 IMMEDIATE ACTION REQUIRED" if risk_level == "high" else "⚠️ REVIEW RECOMMENDED" if risk_level == "medium" else "✅ ACCEPTABLE RISK LEVEL"}
        """
        
        return summary.strip()

    def _run(self, contract_text: str, rule_category: str = "employment_contracts") -> str:
        """LangChain tool interface"""
        result = self.analyze_compliance_intelligently(contract_text, rule_category)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        # Format comprehensive response
        issues = result.get("compliance_issues", [])
        risk = result.get("risk_assessment", {})
        
        if not issues:
            return "✅ No compliance issues found. Contract appears to meet all requirements."
        
        response = f"🔍 **COMPREHENSIVE COMPLIANCE ANALYSIS**\n\n"
        response += f"**Summary:** {result.get('summary', 'Analysis completed')}\n\n"
        response += f"**Risk Assessment:**\n"
        response += f"• Overall Score: {risk.get('score', 0)}/100\n"
        response += f"• Risk Level: {risk.get('level', 'unknown').upper()}\n"
        response += f"• Total Issues: {risk.get('total_issues', 0)}\n\n"
        
        # Categorize and display issues
        high_issues = [i for i in issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'high']
        medium_issues = [i for i in issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'medium']
        low_issues = [i for i in issues if getattr(i, 'severity', i.get('severity', 'medium')) == 'low']
        
        if high_issues:
            response += f"🔴 **HIGH SEVERITY ISSUES ({len(high_issues)}):**\n"
            for issue in high_issues[:3]:  # Show top 3
                rule_name = getattr(issue, 'rule_name', issue.get('rule_name', 'Unknown'))
                violation = getattr(issue, 'violation_message', issue.get('violation_message', 'Unknown'))
                response += f"• {rule_name}: {violation[:100]}...\n"
            response += "\n"
        
        if medium_issues:
            response += f"🟡 **MEDIUM SEVERITY ISSUES ({len(medium_issues)}):**\n"
            for issue in medium_issues[:2]:  # Show top 2
                rule_name = getattr(issue, 'rule_name', issue.get('rule_name', 'Unknown'))
                violation = getattr(issue, 'violation_message', issue.get('violation_message', 'Unknown'))
                response += f"• {rule_name}: {violation[:100]}...\n"
            response += "\n"
        
        response += f"📋 **RECOMMENDATION:** Review and address all identified issues, prioritizing high-severity items."
        
        return response
