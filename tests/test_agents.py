import pytest
from src.agents.cv_analyzer import CVAnalyzerAgent
from src.agents.jd_analyzer import JDAnalyzerAgent
from src.utils.llm_config import LLMConfig

@pytest.fixture
def llm():
    return LLMConfig.get_llm(temperature=0)

@pytest.fixture
def sample_cv():
    return """
    John Doe
    Software Engineer
    
    Skills: Python, Java, Machine Learning, Docker
    Experience: 5 years in software development
    Education: BS Computer Science
    """

@pytest.fixture
def sample_jd():
    return """
    Senior Software Engineer
    
    Required Skills:
    - Python
    - Machine Learning
    - Docker
    
    Experience: 5+ years
    """

def test_cv_analyzer(llm, sample_cv):
    """Test CV analysis"""
    analyzer = CVAnalyzerAgent(llm)
    result = analyzer.analyze(sample_cv)
    
    assert result["success"] is True
    assert "skills" in result["analysis"]
    assert len(result["analysis"]["skills"]) > 0

def test_jd_analyzer(llm, sample_jd):
    """Test JD analysis"""
    analyzer = JDAnalyzerAgent(llm)
    result = analyzer.analyze(sample_jd)
    
    assert result["success"] is True
    assert "job_title" in result["analysis"]
    assert "required_skills" in result["analysis"]

def test_cv_analyzer_error_handling(llm):
    """Test error handling"""
    analyzer = CVAnalyzerAgent(llm)
    result = analyzer.analyze("")
    
    # Should handle empty input gracefully
    assert "error" in result or result["success"] is False