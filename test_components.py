"""
Quick test script to verify all components work.
Run this with: python test_components.py
"""

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.error_handlers import ErrorHandler, UserFriendlyError, ErrorCategory
        print("✅ error_handlers imported")
    except ImportError as e:
        print(f"❌ error_handlers failed: {e}")
        return False
    
    try:
        from utils.llm_normalizer import LLMOutputNormalizer, CodeExtractor
        print("✅ llm_normalizer imported")
    except ImportError as e:
        print(f"❌ llm_normalizer failed: {e}")
        return False
    
    try:
        from validators.code_validators import ValidationPipeline
        print("✅ code_validators imported")
    except ImportError as e:
        print(f"❌ code_validators failed: {e}")
        return False
    
    try:
        from agents.push_agent import PushAgent
        print("✅ push_agent imported")
    except ImportError as e:
        print(f"❌ push_agent failed: {e}")
        return False
    
    try:
        from agents.sub_agents import MultiFileRefactorAgent
        print("✅ sub_agents imported")
    except ImportError as e:
        print(f"❌ sub_agents failed: {e}")
        return False
    
    return True


def test_llm_normalizer():
    """Test LLM output normalization."""
    print("\nTesting LLM normalizer...")
    
    from utils.llm_normalizer import LLMOutputNormalizer
    
    normalizer = LLMOutputNormalizer(provider="vertex_ai")
    
    # Test string input
    result = normalizer.normalize("Hello world")
    assert result.text == "Hello world", "String normalization failed"
    print("✅ String normalization works")
    
    # Test list input
    result = normalizer.normalize([{"text": "Test output"}])
    assert result.text == "Test output", "List normalization failed"
    print("✅ List normalization works")
    
    # Test dict input
    result = normalizer.normalize({"text": "Dict output"})
    assert result.text == "Dict output", "Dict normalization failed"
    print("✅ Dict normalization works")
    
    return True


def test_code_extractor():
    """Test code extraction."""
    print("\nTesting code extractor...")
    
    from utils.llm_normalizer import CodeExtractor
    
    extractor = CodeExtractor()
    
    # Test markdown fence removal
    code_with_fence = "```python\nprint('hello')\n```"
    cleaned = extractor.remove_markdown_fences(code_with_fence)
    assert "```" not in cleaned, "Fence removal failed"
    print("✅ Markdown fence removal works")
    
    # Test preamble removal
    code_with_preamble = "Here's the code:\nprint('hello')"
    cleaned = extractor.remove_ai_preamble(code_with_preamble)
    assert "Here's" not in cleaned, "Preamble removal failed"
    print("✅ Preamble removal works")
    
    return True


def test_push_agent_extraction():
    """Test file extraction from AI response."""
    print("\nTesting PushAgent file extraction...")
    
    from agents.push_agent import PushAgent
    
    agent = PushAgent()
    
    # Test with proper markers
    response = """// === test.js ===
function hello() {
    console.log('world');
}

// === utils.js ===
export const add = (a, b) => a + b;
"""
    
    files = agent.extract_files_from_response(response)
    assert len(files) == 2, f"Expected 2 files, got {len(files)}"
    assert "test.js" in files, "test.js not found"
    assert "utils.js" in files, "utils.js not found"
    print(f"✅ Extracted {len(files)} files correctly")
    
    return True


def test_validator():
    """Test code validation."""
    print("\nTesting code validator...")
    
    from validators.code_validators import CodeValidator
    
    validator = CodeValidator()
    
    # Test valid Python code
    valid_code = "def hello():\n    return 'world'"
    result = validator.validate_file("test.py", valid_code, "python")
    assert result.is_valid, "Valid code marked as invalid"
    print("✅ Valid code detection works")
    
    # Test invalid Python code
    invalid_code = "def hello(\n    return 'world'"
    result = validator.validate_file("test.py", invalid_code, "python")
    assert not result.is_valid, "Invalid code marked as valid"
    assert len(result.errors) > 0, "No errors detected for invalid code"
    print("✅ Invalid code detection works")
    
    return True


def test_error_handler():
    """Test error handler."""
    print("\nTesting error handler...")
    
    from utils.error_handlers import ErrorHandler, ErrorCategory
    
    handler = ErrorHandler()
    
    # Test auth error handling
    error = handler.handle_github_auth_error(Exception("Invalid token"))
    assert error.category == ErrorCategory.GITHUB_AUTH
    assert len(error.suggested_actions) > 0
    print("✅ Auth error handling works")
    
    # Test rate limit error handling
    error = handler.handle_rate_limit_error(Exception("Rate limit exceeded"))
    assert error.category == ErrorCategory.RATE_LIMIT
    assert error.can_retry
    print("✅ Rate limit error handling works")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("REPO AI Component Tester")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("LLM Normalizer", test_llm_normalizer),
        ("Code Extractor", test_code_extractor),
        ("PushAgent Extraction", test_push_agent_extraction),
        ("Code Validator", test_validator),
        ("Error Handler", test_error_handler),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your components are ready.")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Load a repository")
        print("3. Try a refactoring request")
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before running the app.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)