from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List, Dict
import operator

# Définition de l'état partagé
class InterviewPrepState(TypedDict):
    cv_text: str
    cv_analysis: Dict
    jd_text: str
    jd_analysis: Dict
    company_name: str
    company_info: Dict
    questions: List[Dict]
    current_question_idx: int
    user_answers: List[Dict]
    feedback_history: List[Dict]
    general_tips: Dict
    human_approval_needed: bool
    human_feedback: str
    next_step: str
    error: str

class InterviewPrepSupervisor:
    def __init__(self, agents: Dict, vector_store, memory_saver):
        self.agents = agents
        self.vector_store = vector_store
        self.memory = memory_saver
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Construit le graphe LangGraph"""
        workflow = StateGraph(InterviewPrepState)
        
        # Ajouter les nœuds
        workflow.add_node("analyze_cv", self.analyze_cv_node)
        workflow.add_node("analyze_jd", self.analyze_jd_node)
        workflow.add_node("research_company", self.research_company_node)
        workflow.add_node("generate_questions", self.generate_questions_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("conduct_interview", self.conduct_interview_node)
        workflow.add_node("provide_feedback", self.provide_feedback_node)
        workflow.add_node("generate_tips", self.generate_tips_node)
        
        # Définir le flux
        workflow.set_entry_point("analyze_cv")
        
        workflow.add_edge("analyze_cv", "analyze_jd")
        workflow.add_edge("analyze_jd", "research_company")
        workflow.add_edge("research_company", "generate_questions")
        workflow.add_edge("generate_questions", "human_review")
        
        # Conditional edge après human review
        workflow.add_conditional_edges(
            "human_review",
            self.route_after_human_review,
            {
                "approved": "generate_tips",
                "regenerate": "generate_questions",
                "interview": "conduct_interview"
            }
        )
        
        workflow.add_edge("generate_tips", "conduct_interview")
        workflow.add_edge("conduct_interview", "provide_feedback")
        
        # Conditional edge pour continuer ou terminer
        workflow.add_conditional_edges(
            "provide_feedback",
            self.route_after_feedback,
            {
                "continue": "conduct_interview",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory)
    
    def analyze_cv_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'analyse CV"""
        result = self.agents["cv_analyzer"].analyze(state["cv_text"])
        
        if result["success"]:
            # Stocker dans vector DB
            self.vector_store.add_documents(
                "cv_data",
                [state["cv_text"]],
                [{"type": "cv", "analysis": result["analysis"]}]
            )
        
        return {
            **state,
            "cv_analysis": result["analysis"],
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def analyze_jd_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'analyse JD"""
        result = self.agents["jd_analyzer"].analyze(state["jd_text"])
        
        if result["success"]:
            # Stocker dans vector DB
            self.vector_store.add_documents(
                "jd_data",
                [state["jd_text"]],
                [{"type": "jd", "analysis": result["analysis"]}]
            )
        
        return {
            **state,
            "jd_analysis": result["analysis"],
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def research_company_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de recherche entreprise"""
        industry = state["jd_analysis"].get("industry", "")
        result = self.agents["company_researcher"].research(
            state["company_name"],
            industry
        )
        
        return {
            **state,
            "company_info": result.get("info", {}),
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def generate_questions_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de génération de questions"""
        result = self.agents["question_generator"].generate_questions(
            state["cv_analysis"],
            state["jd_analysis"],
            state["company_info"]
        )
        
        return {
            **state,
            "questions": result.get("questions", []),
            "current_question_idx": 0,
            "human_approval_needed": True,
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def human_review_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'attente de validation humaine"""
        # Ce nœud met en pause le workflow pour intervention humaine
        return {
            **state,
            "next_step": "awaiting_human_input"
        }
    
    def conduct_interview_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de simulation d'entretien"""
        # Retourne l'état avec la question courante
        # L'interaction réelle se fait dans l'UI
        return state
    
    def provide_feedback_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de feedback sur une réponse"""
        if not state["user_answers"]:
            return state
        
        last_answer = state["user_answers"][-1]
        question = state["questions"][last_answer["question_idx"]]
        
        context = {
            "cv": state["cv_analysis"],
            "jd": state["jd_analysis"]
        }
        
        result = self.agents["interview_coach"].evaluate_answer(
            question["question"],
            last_answer["answer"],
            context
        )
        
        feedback_history = state.get("feedback_history", [])
        if result["success"]:
            feedback_history.append({
                "question_idx": last_answer["question_idx"],
                "feedback": result["feedback"]
            })
        
        return {
            **state,
            "feedback_history": feedback_history,
            "current_question_idx": state["current_question_idx"] + 1
        }
    
    def generate_tips_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de génération de conseils généraux"""
        result = self.agents["interview_coach"].generate_general_tips(
            state["cv_analysis"],
            state["jd_analysis"]
        )
        
        return {
            **state,
            "general_tips": result.get("tips", {})
        }
    
    def route_after_human_review(self, state: InterviewPrepState) -> str:
        """Détermine la route après validation humaine"""
        feedback = state.get("human_feedback", "approved")
        
        if feedback == "regenerate":
            return "regenerate"
        elif feedback == "interview":
            return "interview"
        else:
            return "approved"
    
    def route_after_feedback(self, state: InterviewPrepState) -> str:
        """Détermine si on continue l'entretien ou on termine"""
        if state["current_question_idx"] >= len(state["questions"]):
            return "end"
        return "continue"