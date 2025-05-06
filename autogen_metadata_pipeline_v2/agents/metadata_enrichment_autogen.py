"""AutoGen AssistantAgent for enriching metadata."""
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from autogen_core.models import UserMessage       # <- NEW import

# ────────────  TOOL FUNCTIONS ────────────
from typing import List, Dict, Any
from autogen_ext.models.openai import OpenAIChatCompletionClient  # works for any api_type
client = OpenAIChatCompletionClient(model="gemini-1.5-flash", api_key="")  # <- NEW
def load_discovery() -> List[Dict[str, Any]]:
    """Read discovery_output.json and return its list."""
    with open("discovery_output.json") as f:
        return json.load(f)

import re, json
from typing import Any

# ── NEW helper ───────────────────────────────────────────────────────
def _to_str(resp: Any) -> str:
    """
    Normalise any LLM response (CreateResult, SDK object, dict, str) → str.
    """
    # 1) plain string already
    if isinstance(resp, str):
        return resp

    # 2) google-generativeai SDK
    if hasattr(resp, "text"):
        return resp.text

    # 3) OpenAI-style dict
    if isinstance(resp, dict) and "choices" in resp:
        return resp["choices"][0]["message"]["content"]

    # 4) AutoGen CreateResult (pydantic)
    if hasattr(resp, "choices"):
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            return choice.message.content

    # fallback
    return str(resp)


# ── JSON-extractor (unchanged API, just calls _to_str first) ─────────
def _extract_json(raw_resp: Any) -> str:
    """
    Pull the first JSON object/array from an LLM reply.
    Accepts str, dict, CreateResult, or SDK object.
    """
    raw_text = _to_str(raw_resp)
    raw = raw_text.split("```", 1)[-1]

    # strip ```json fences
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.S | re.I)
    if fenced:
        raw_text = fenced.group(1)

    # grab first {...} or [...]
    match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.S)
    return match.group(1).strip() if match else raw_text.strip()


def enrich(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gemini adds:
      { "description": str, "category": str }
    and we merge that back into the original entry.
    """
    prompt = (
                    f"Given the following metadata field: {entry}\n"
                    f"- Generate a meaningful description.\n"
                    f"- Identify and mention its semantic category (e.g., \"Price\", \"Review Data\", \"Product Details\").\n"
                    f"- Suggest any missing values.\n"
                )

    async def _query_llm() -> Dict[str, Any]:
        resp = await client.create(messages=[UserMessage(content=prompt, 
                                                         source="enricher")] 
)
        print(resp)
        json_str  = _extract_json(resp)                   # ← NEW

        try:
            enrich_obj: dict[str, Any] = json.loads(json_str)  # ← unchanged
        except json.JSONDecodeError:
            enrich_obj = {"description": json_str, "category": "unknown"}
        return {**entry, **enrich_obj}

    enrich_obj: Dict[str, Any] = asyncio.run(_query_llm())
    return {**entry, **enrich_obj}


def store_enriched(payload: List[Dict[str, Any]]) -> str:
    with open("enriched_output.json", "w") as f:
        json.dump(payload, f, indent=2)
    return "✅ enriched_output.json saved"

# ──────────── AGENT ────────────
model = OpenAIChatCompletionClient(model="gemini-1.5-flash",api_key="AIzaSyCMYKN9GbBtPfuo03mteeqw8HLAIV5rGc0")
enrichment_agent = AssistantAgent(
    "enricher",
    system_message="You add .",
    model_client=model,
    tools=[FunctionTool(load_discovery,description="Load discovery output"),
           FunctionTool(enrich,description="Enrich metadata from discovered metadata from PDF Files"),
           FunctionTool(store_enriched,description="Store enriched metadata")],
    reflect_on_tool_use=True,
)
