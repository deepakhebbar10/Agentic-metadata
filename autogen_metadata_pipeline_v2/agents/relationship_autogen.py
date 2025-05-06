"""AutoGen AssistantAgent that infers relationships between enriched fields."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# --- autogen imports ---------------------------------------------------------
from autogen_core.models import UserMessage
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent          # keep your path
from autogen_ext.models.openai import OpenAIChatCompletionClient

# --- Gemini via OpenAIChatCompletionClient -----------------------------------
# (api_type="google" is implied by model name “gemini-*” in recent pyautogen)
client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    api_key="AIzaSyCMYKN9GbBtPfuo03mteeqw8HLAIV5rGc0",             # ⚠️ replace or pull from env
)

# -----------------------------------------------------------------------------
# 📄 helpers
# -----------------------------------------------------------------------------
def _to_text(resp: Any) -> str:
    """Flatten CreateResult / dict / SDK object to plain text."""
    if hasattr(resp, "choices"):                     # CreateResult
        return resp.choices[0].message.content
    if isinstance(resp, dict) and "choices" in resp: # OpenAI dict form
        return resp["choices"][0]["message"]["content"]
    if hasattr(resp, "text"):                        # google-generativeai
        return resp.text
    return str(resp)


def _extract_json(resp: Any) -> str:
    """
    Strip ```json fences and return the first {...} or [...] block
    from the LLM reply.
    """
    raw = _to_text(resp)

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S | re.I)
    if fenced:
        raw = fenced.group(1)

    block = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    return block.group(1).strip() if block else raw.strip()


# -----------------------------------------------------------------------------
# 📄  FunctionTools (all fully typed)
# -----------------------------------------------------------------------------
ENRICHED_FILE = Path("enriched_output.json")
RELATIONS_FILE = Path("relationships_output.json")


def load_enriched() -> List[Dict[str, Any]]:
    """Load enriched_output.json produced by the previous agent."""
    return json.loads(ENRICHED_FILE.read_text())


def infer_relationships(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ask Gemini for relationships between enriched PDF entries.
    Returns a JSON list with objects like:
      { "source": "...", "target": "...", "type": "...", "comment": "..." }
    """

    prompt = f"""
You are a metadata analysis AI.

The target field is src⁠, with the enriched description:


Please:
1.⁠ ⁠Determine which of these fields are related to ⁠src.
2.⁠ ⁠For each related field, state the file name or database where it appears.
3.⁠ ⁠In one concise sentence each, describe the nature of the relationship (e.g., same category, similar price range, related product, shared discount patterns).

Return your response as valid JSON, where each key is a related field name and its value is an object with:
•⁠  ⁠⁠ "file" ⁠: the source file or database name  
•⁠  ⁠⁠ "relationship" ⁠: a one-sentence description of their connection  
"""

    async def _query_llm() -> List[Dict[str, Any]]:
        resp = await client.create(
            messages=[UserMessage(content=prompt, source="relator")]
        )

        json_str  = _extract_json(resp) 
        json_str = json_str.replace("\\'", "'")
        try:
            return json.loads(_extract_json(resp))
        except json.JSONDecodeError:
            return [{"comment": _to_text(resp).strip()}]

    # run coroutine synchronously (safe on Python 3.13 with a fresh loop)
    return asyncio.run(_query_llm())


def store_rels(rels: List[Dict[str, Any]]) -> str:
    RELATIONS_FILE.write_text(json.dumps(rels, indent=2))
    return "✅ relationships_output.json saved"


# -----------------------------------------------------------------------------
# 📄  AssistantAgent
# -----------------------------------------------------------------------------
relationship_agent = AssistantAgent(
    name="relator",
    system_message="You are an AI assistant that infers relationships between metadata fields." ,
    model_client=client,          # OpenAIChatCompletionClient instance
    tools=[
        FunctionTool(load_enriched,        description="Load enriched metadata"),
        FunctionTool(infer_relationships,  description="Derive relationships"),
        FunctionTool(store_rels,           description="Store relationships"),
    ],
    reflect_on_tool_use=True,
)
