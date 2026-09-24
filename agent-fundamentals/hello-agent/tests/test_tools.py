import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import calculate_expression, validate_arguments


def test_calculate_expression_basic():
    assert calculate_expression("22 * 5") == "Result: 110"


def test_calculate_expression_rejects_unsafe():
    # Only arithmetic AST node types are allowed; names/calls are never reached.
    result = calculate_expression("__import__('os').system('echo hi')")
    assert result.startswith("Error evaluating expression")


def test_validate_arguments_missing_field():
    assert validate_arguments("get_weather", {}) == "missing required argument 'location'"


def test_validate_arguments_wrong_type():
    err = validate_arguments("calculate_expression", {"expression": 5})
    assert err is not None and "must be a string" in err


def test_validate_arguments_unknown_tool():
    assert validate_arguments("delete_database", {}) == "unknown tool 'delete_database'"


def test_validate_arguments_valid():
    assert validate_arguments("get_weather", {"location": "Tokyo"}) is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")