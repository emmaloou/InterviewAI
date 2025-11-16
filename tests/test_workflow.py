import pytest
from src.agents.supervisor import InterviewPrepSupervisor
from src.utils.llm_config import LLMConfig
from langgraph.checkpoint.sqlite import SqliteSaver

@pytest.fixture
def supervisor():
    llm = LLMConfig.get_llm()
    # Initialize all agents and tools
    # ... (simplified for brevity)
    memory = SqliteSaver.from_conn_string(":memory:")
    return InterviewPrepSupervisor(agents, vector_store, memory)

def test_full_workflow(supervisor):
    """Test complete workflow"""
    initial_state = {
        "cv_text": "Sample CV content",
        "jd_text": "Sample JD content",
        "company_name": "Test Company",
        # ... other fields
    }
    
    config = {"configurable": {"thread_id": "test_1"}}
    
    # Execute workflow
    final_state = None
    for state in supervisor.graph.stream(initial_state, config):
        final_state = list(state.values())[0]
    
    assert final_state is not None
    assert "questions" in final_state
    assert len(final_state["questions"]) > 0