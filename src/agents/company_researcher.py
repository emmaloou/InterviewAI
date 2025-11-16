from typing import Dict, List
import json

class CompanyResearcherAgent:
    def __init__(self, llm, web_search_tool):
        self.llm = llm
        self.web_search = web_search_tool
        
    def research(self, company_name: str, industry: str = "") -> Dict:
        """Recherche des informations sur une entreprise"""
        # 1. Recherche web
        search_results = self.web_search.search_company_info(company_name, industry)
        
        if not search_results["success"]:
            return {
                "success": False,
                "error": "Web search failed",
                "info": {}
            }
        
        # 2. Synthèse avec LLM
        results_text = "\n\n".join([
            f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
            for r in search_results["results"][:3]
        ])
        
        prompt = f"""Synthétise les informations suivantes sur l'entreprise {company_name}:

{results_text}

Retourne un JSON avec:
- company_name: nom
- main_activity: activité principale
- recent_news: actualités récentes (liste de 2-3 points)
- company_culture: culture d'entreprise
- values: valeurs (liste)
- industry_challenges: défis du secteur (liste)
- interesting_facts: faits intéressants (liste)

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            info = json.loads(response)
            
            return {
                "success": True,
                "info": info,
                "sources": [r.get("url") for r in search_results["results"][:3]]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "info": {}
            }