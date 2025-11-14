"""AI + heuristic Postman collection generator used by app_streamlit.py.

Public API:
    generate_postman_collection(swagger: dict, divergences: list, use_llm: bool=False,
                                model: str="gpt-4o-mini", gateway_url: Optional[str]=None) -> dict
If gateway_url is None and use_llm is True, this module will use the QBURST gateway:
    https://llmgateway.qburst.build/v1
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

# Try to detect OpenAI SDK availability (new and old)
_HAS_OPENAI = False
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    try:
        import openai  # type: ignore
        _HAS_OPENAI = True
    except Exception:
        _HAS_OPENAI = False

# Default QBurst gateway (used when use_llm True and no gateway_url provided)
_QBURST_DEFAULT_GATEWAY = "https://llmgateway.qburst.build/v1"


def _safe_json_load(text: str):
    """Try to parse JSON; if wrapped, extract first {...} block."""
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)
        raise


def _heuristic_examples_from_schema(schema: Any):
    """Make quick positive/negative/edge examples from a swagger-like schema dict."""
    positive = {}
    negative = {}
    edge = {}
    if not isinstance(schema, dict):
        return {"positive": {"sample": "value"}, "negative": {"missing": None}, "edge": {"sample": ""}}
    props = schema.get("properties", {})
    required = schema.get("required", []) if isinstance(schema.get("required", []), list) else []
    for k, v in props.items():
        t = v.get("type") if isinstance(v, dict) else "string"
        if t == "string":
            positive[k] = f"example_{k}"
            negative[k] = ""
            edge[k] = "x" * 300
        elif t in ("integer", "number"):
            positive[k] = 1
            negative[k] = "not_a_number"
            edge[k] = 999999999
        elif t == "boolean":
            positive[k] = True
            negative[k] = "not_bool"
            edge[k] = False
        else:
            positive[k] = None
            negative[k] = None
            edge[k] = None
    for r in required:
        if r not in positive:
            positive[r] = f"example_{r}"
    return {"positive": positive, "negative": negative, "edge": edge}


def _build_postman_item(name: str, method: str, url: str, body: Optional[dict] = None,
                       tests: Optional[List[str]] = None):
    item = {
        "name": name,
        "request": {
            "method": method.upper(),
            "header": [],
            "url": {
                "raw": "{{base_url}}" + url,
                "host": ["{{base_url}}"],
                "path": url.lstrip("/").split("/") if url else []
            }
        }
    }
    if body is not None:
        item["request"]["body"] = {
            "mode": "raw",
            "raw": json.dumps(body, indent=2),
            "options": {"raw": {"language": "json"}}
        }
    if tests:
        item["event"] = [{
            "listen": "test",
            "script": {"type": "text/javascript", "exec": tests}
        }]
    return item


def _heuristic_collection(divergences: List[Dict[str, Any]]):
    """Fallback heuristic generator if LLM not available or fails."""
    coll = {
        "info": {
            "name": "Auto-generated Tests - Divergence Report (Heuristic)",
            "_postman_id": str(int(time.time())),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }
    for d in divergences:
        ep = d.get("endpoint", "")
        method = (d.get("method") or "GET").upper()
        path = ep.split(" ", 1)[1] if " " in ep else ep
        schema = d.get("schema") or d.get("expected") or {}
        examples = _heuristic_examples_from_schema(schema if isinstance(schema, dict) else {})
        pos = examples.get("positive")
        neg = examples.get("negative")
        edge = examples.get("edge")

        pos_test = ["pm.test('Status is 2xx', function(){pm.expect(pm.response.code).to.be.within(200,299);});"]
        neg_test = ["pm.test('Status is 4xx or 5xx for invalid input', function(){pm.expect(pm.response.code).to.be.within(400,599);});"]
        edge_test = ["pm.test('Edge case handled', function(){pm.expect(pm.response.code).to.be.within(200,499);});"]

        if pos:
            coll["item"].append(_build_postman_item(f"{method} {path} - Positive", method, path, body=pos, tests=pos_test))
        if neg:
            coll["item"].append(_build_postman_item(f"{method} {path} - Negative", method, path, body=neg, tests=neg_test))
        if edge:
            coll["item"].append(_build_postman_item(f"{method} {path} - Divergence", method, path, body=edge, tests=edge_test))
    return coll


def _call_llm_generate(swagger: Dict[str, Any], divergences: List[Dict[str, Any]],
                       model: str = "gpt-4o-mini", temperature: float = 0.2,
                       gateway_url: Optional[str] = None):
    """
    Call the LLM (supports both new OpenAI SDK and legacy openai).
    Returns parsed JSON dict (Postman collection) or raises on failure.
    """
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI SDK not installed. Install 'openai' to enable LLM mode.")

    # Enhanced, structured system prompt
    system_prompt = (
        "You are an expert QA automation engineer and API testing specialist.\n\n"
        "You are given two inputs:\n"
        "1. A Swagger API specification.\n"
        "2. A divergence report listing mismatches between the Swagger spec and actual code.\n\n"
        "Your goal:\n"
        "- For EACH issue in the divergence report, create a dedicated **folder** inside the Postman collection.\n"
        "- Within that folder, generate multiple **test cases (requests)** categorized as:\n"
        "  - Positive Tests (expected behavior based on Swagger)\n"
        "  - Negative Tests (invalid inputs, missing params, unauthorized access, etc.)\n"
        "  - Divergence Tests (tests focused on the issue described in the divergence report)\n\n"
        "Mandatory Rule:\n"
        "Every folder in the Postman collection MUST contain at least:\n"
        "- One Positive test case\n"
        "- One Negative test case\n"
        "- One Divergence test case\n"
        "Even if the divergence report does not explicitly mention testable details, generate representative test scenarios for all three categories.\n\n"
        "Output:\n"
        "Return ONLY a valid Postman Collection JSON (v2.1 schema). Use {{base_url}} for host. Include basic pm.test() assertions for response status and validation."
    )

    payload = {"swagger": swagger, "divergences": divergences}

    # if gateway_url not provided, use QBurst default
    if not gateway_url:
        gateway_url = _QBURST_DEFAULT_GATEWAY

    messages_payload = json.dumps(payload)

    # Try new OpenAI client first
    try:
        from openai import OpenAI as OpenAIClient  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or API_KEY not set in environment for LLM gateway.")
        client = OpenAIClient(api_key=api_key, base_url=gateway_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_payload}
            ],
            temperature=temperature
        )
        text = ""
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = getattr(resp.choices[0], "text", "")
        return _safe_json_load(text)
    except Exception:
        # fallback legacy
        try:
            import openai  # type: ignore
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY or API_KEY not set in environment for LLM gateway.")
            if gateway_url:
                openai.api_base = gateway_url
            openai.api_key = api_key
            client = OpenAIClient(api_key=api_key, base_url=gateway_url)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages_payload}
                ],
                temperature=temperature
            )
            text = ""
            try:
                text = resp.choices[0].message['content']
            except Exception:
                text = getattr(resp.choices[0], "text", "")
            return _safe_json_load(text)
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")


def generate_postman_collection(swagger: Dict[str, Any], divergences: List[Dict[str, Any]],
                                use_llm: bool = False, model: str = "gpt-4o-mini",
                                gateway_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Public entrypoint used by Streamlit app.
    """
    if use_llm:
        try:
            col = _call_llm_generate(swagger, divergences, model=model, gateway_url=gateway_url)
            if not isinstance(col, dict):
                raise RuntimeError("LLM returned non-dict output")
            return col
        except Exception as e:
            print("LLM generation failed, falling back to heuristic. Error:", e)
            return _heuristic_collection(divergences)
    else:
        return _heuristic_collection(divergences)
