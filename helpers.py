import re


def build_prompt(context: str, pytest_output: list, target_filename: str) -> list:
    """
    Builds the shared debugging prompt template used for both control and full-context runs.
    Returns a list of message dicts with proper system/user role separation.
    """
    failures_str = "\n".join([str(f) for f in pytest_output])

    system_content = (
        "You are an expert Python debugger. Given a large multi-file Python codebase\n"
        "and failing test output, find and fix the single bug. Return ONLY the corrected\n"
        "version of the file that contains the bug inside a ```python block.\n"
        "Do not modify other files. Do not explain."
    )

    user_content = (
        f"--- FAILING TEST OUTPUT ---\n"
        f"{failures_str}\n"
        f"\n"
        f"--- REPOSITORY SOURCE ---\n"
        f"{context}\n"
        f"\n"
        f"Return the corrected file: {target_filename}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

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
    Estimates token footprint using tiktoken's cl100k_base encoding.
    Falls back to whitespace word count if tiktoken is unavailable.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        print("[WARNING] tiktoken not available — using word-count approximation")
        return len(text.split())