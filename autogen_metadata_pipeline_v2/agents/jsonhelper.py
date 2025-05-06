import re, json
from typing import Any, List, Dict

def parse_llm_blob(blob: str) -> List[Dict[str, Any]]:
    """
    Extract the JSON array from a Gemini / OpenAI debug blob like:

        finish_reason='stop' content='```json
        [  {...}, {...} ]
        ```
        ' usage=RequestUsage(...)

    Returns a Python list of dicts.
    """

    # 1️⃣ pull out the stuff between  content='  and  ' usage=
    content_match = re.search(r"content='(.*?)'\s+usage=", blob, re.S)
    if not content_match:
        raise ValueError("Cannot locate content='…' segment")

    raw = content_match.group(1)

    # 2️⃣ strip ```json fences (or plain ```) if present
    if "```" in raw:
        raw = raw.split("```", 1)[-1]        # drop leading
        raw = raw.split("```", 1)[0]         # drop trailing

    # 3️⃣ unescape invalid  \\'   produced by Gemini
    raw = raw.replace("\\'", "'")

    # 4️⃣ parse – falls back to literal_eval if JSON still invalid
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import ast
        return ast.literal_eval(raw)

# --------------------------------------------------------------------

