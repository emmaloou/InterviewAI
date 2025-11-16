from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict
import json

class CVAnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Tu es un expert en analyse de CV.
            
            Analyse ce CV et extrais les informations clés:
            {cv_text}
            
            Retourne un JSON avec:
            - skills: liste des compétences
            - experience_years: nombre d'années d'expérience
            - experience_domains: domaines d'expertise
            - education: formation
            - strengths: 3-5 points forts
            - areas_for_improvement: 2-3 points à améliorer
            - summary: résumé en 2-3 phrases
            
            JSON:"""
        )
    
    def analyze(self, cv_text: str) -> Dict:
        """Analyse un CV"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"cv_text": cv_text})
            
            # Parse JSON response
            analysis = json.loads(response)
            
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }