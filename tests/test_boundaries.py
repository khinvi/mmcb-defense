import sys
import os
import pytest
sys.path.insert(0, os.path.abspath('src'))
from boundaries.token_boundary import TokenBoundary
from boundaries.semantic_boundary import SemanticBoundary
from boundaries.hybrid_boundary import HybridBoundary

def test_token_boundary():
    tb = TokenBoundary()
    text = tb.apply_text_boundary("test system", is_system=True)
    assert isinstance(text, str) and "SYSTEM_INSTRUCTION_BEGIN" in text
    file = tb.apply_file_boundary("file content", file_type="json")
    assert isinstance(file, str) and "FILE_CONTENT_BEGIN" in file

def test_semantic_boundary():
    sb = SemanticBoundary()
    prompt = sb.apply_boundary("sys", "user", "file content")
    assert isinstance(prompt, str)
    assert "Priority Level" in prompt or "system" in prompt.lower()

def test_hybrid_boundary():
    hb = HybridBoundary()
    prompt = hb.apply_boundary("sys", "user", "file content")
    assert isinstance(prompt, str)
    assert ("SYSTEM_INSTRUCTION_BEGIN" in prompt or "Priority Level" in prompt)
