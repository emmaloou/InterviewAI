from tavily import TavilyClient
import os
from typing import List, Dict

class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def search_company_info(self, company_name: str, additional_context: str = "") -> Dict:
        """Recherche des informations sur une entreprise"""
        query = f"{company_name} company information culture values {additional_context}"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            return {
                "success": True,
                "results": response.get("results", []),
                "query": query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def search_interview_tips(self, job_title: str, industry: str) -> Dict:
        """Recherche des conseils d'entretien pour un poste"""
        query = f"interview tips {job_title} {industry} best practices"
        
        try:
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            return {
                "success": True,
                "results": response.get("results", []),
                "query": query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }