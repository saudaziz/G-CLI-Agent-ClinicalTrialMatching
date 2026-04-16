import subprocess
import os
import sys

def run_app(lang):
    if lang == "python":
        print("Running Python implementation...")
        result = subprocess.run(
            ["python", "main.py"],
            cwd="agent-system",
            capture_output=True,
            text=True
        )
        return result.stdout
    elif lang == "csharp":
        print("Running C# implementation...")
        result = subprocess.run(
            ["dotnet", "run"],
            cwd="agent-system-cs",
            capture_output=True,
            text=True
        )
        return result.stdout
    return ""

def test_p001_match(lang):
    output = run_app(lang)
    # Check for core justification markers
    assert "MATCH" in output.upper()
    assert "P001" in output
    print(f"{lang.capitalize()} Test Passed: P001 matches correctly.")

if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "python"
    try:
        test_p001_match(lang)
    except Exception as e:
        print(f"{lang.capitalize()} test failed: {e}")
        sys.exit(1)
