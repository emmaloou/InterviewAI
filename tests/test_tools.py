import pytest
from src.tools.document_parser import DocumentParser
from pathlib import Path

def test_parse_text_file(tmp_path):
    """Test parsing text file"""
    # Create temporary file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    result = DocumentParser.parse_document(str(test_file))
    
    assert result["content"] == "Test content"
    assert result["extension"] == "txt"

def test_unsupported_format():
    """Test unsupported file format"""
    with pytest.raises(ValueError):
        DocumentParser.parse_document("test.xyz")