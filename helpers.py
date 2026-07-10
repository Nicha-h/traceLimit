import re


def build_prompt(context: str, pytest_output: list, target_filename: str) -> str:
    """
    Builds the shared debugging prompt template used for both control and full-context runs.
    """
    failures_str = "\n".join([str(f) for f in pytest_output])

    return f"""SYSTEM:
You are an expert Python debugger. Given a large multi-file Python codebase
and failing test output, find and fix the single bug. Return ONLY the corrected
version of the file that contains the bug inside a ```python block.
Do not modify other files. Do not explain.

USER:
--- FAILING TEST OUTPUT ---
{failures_str}

--- REPOSITORY SOURCE ---
{context}

Return the corrected file: {target_filename}
"""

def extract_code_block(response_text: str) -> str:
    """
    Extracts raw contents from inside a markdown ```python ... ``` block.
    """
    match = re.search(r"```python\s+(.*?)\s+```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback to returning raw text if the block wrappers are missing
    return response_text.strip()

def count_tokens(text: str) -> int:
    """
    Estimates token footprint. For basic structural logic calculation, 
    a rough word/char count scaling factor works prior to tiktoken binding.
    """
    return len(text.split())