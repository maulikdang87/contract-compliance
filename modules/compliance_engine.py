import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ComplianceIssue:
    """Represents a compliance violation found in a contract"""
    rule_id: str
    rule_name: str
    severity: str
    violation_message: str
    found_keywords: List[str]
    missing_elements: List[str]
    confidence_score: float

class ComplianceEngine:
    def __init__(self, rules_directory: str = "data/rules"):
        """
        Initialize the compliance engine with rule definitions.
        
        Think of this as loading your legal playbook - all the rules
        your AI needs to know to check contracts properly.
        """
        self.rules_directory = Path(rules_directory)
        self.loaded_rules = {}
        self.load_all_rules()
    
    def load_all_rules(self):
        """Load all rule files from the rules directory"""
        for rule_file in self.rules_directory.glob("*.json"):
            with open(rule_file, 'r') as f:
                rule_data = json.load(f)
                category = rule_data["rule_category"]
                self.loaded_rules[category] = rule_data["rules"]
    
    def check_compliance(self, contract_text: str, rule_category: str = "employment_contracts") -> List[ComplianceIssue]:
        """
        Main compliance checking method.
        
        This is where the magic happens - we systematically check
        each rule against the contract text and build a list of issues.
        """
        issues = []
        contract_lower = contract_text.lower()
        
        if rule_category not in self.loaded_rules:
            return issues
        
        for rule in self.loaded_rules[rule_category]:
            issue = self._check_single_rule(contract_lower, rule)
            if issue:
                issues.append(issue)
        
        return issues
    
    def _check_single_rule(self, contract_text: str, rule: Dict) -> ComplianceIssue:
        """
        Check a single rule against contract text.
        
        This is the core logic - for each rule, we:
        1. Look for required keywords
        2. Assess if required elements are present
        3. Calculate confidence in our assessment
        """
        found_keywords = []
        missing_elements = []
        
        # Check for required keywords
        for keyword in rule["keywords"]:
            if keyword.lower() in contract_text:
                found_keywords.append(keyword)
        
        # If no keywords found, this rule likely doesn't apply
        if not found_keywords:
            return None
        
        # Check for required elements
        for element in rule["required_elements"]:
            if element.lower() not in contract_text:
                missing_elements.append(element)
        
        # Calculate confidence score (0-1)
        keyword_score = len(found_keywords) / len(rule["keywords"])
        element_score = 1 - (len(missing_elements) / len(rule["required_elements"]))
        confidence = (keyword_score + element_score) / 2
        
        # Only report as issue if we have missing elements
        if missing_elements:
            return ComplianceIssue(
                rule_id=rule["rule_id"],
                rule_name=rule["rule_name"],
                severity=rule["severity"],
                violation_message=rule["violation_message"],
                found_keywords=found_keywords,
                missing_elements=missing_elements,
                confidence_score=confidence
            )
        
        return None
    
    def calculate_risk_score(self, issues: List[ComplianceIssue]) -> Dict[str, Any]:
        """
        Calculate overall risk assessment.
        
        This gives us a quantitative way to assess how risky
        a contract is from a compliance perspective.
        """
        if not issues:
            return {"score": 0, "level": "low", "total_issues": 0}
        
        # Weight severity levels
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        
        total_weighted_score = sum(
            severity_weights.get(issue.severity, 1) * issue.confidence_score 
            for issue in issues
        )
        
        # Normalize to 0-100 scale
        max_possible_score = len(issues) * 3  # All high severity
        risk_score = (total_weighted_score / max_possible_score) * 100
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "score": round(risk_score, 1),
            "level": risk_level,
            "total_issues": len(issues),
            "high_severity": len([i for i in issues if i.severity == "high"]),
            "medium_severity": len([i for i in issues if i.severity == "medium"]),
            "low_severity": len([i for i in issues if i.severity == "low"])
        }
