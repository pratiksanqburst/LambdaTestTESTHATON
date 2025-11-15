# src/ai_testgen.py
"""
AI + heuristic Postman collection generator used by app_streamlit.py.

Public API:
    generate_postman_collection(swagger: dict, divergences: list, use_llm: bool=False,
                                model: str="gpt-4o-mini", gateway_url: Optional[str]=None) -> dict

If gateway_url is None and use_llm is True, this module will use the QBURST gateway:
    https://llmgateway.qburst.build/v1
"""
from typing import List, Dict, Any, Optional
import os
import json
import time

# Try to detect OpenAI SDK availability (new and old)
_HAS_OPENAI = False
try:
    from openai import OpenAI  # new SDK
    _HAS_OPENAI = True
except Exception:
    try:
        import openai  # legacy
        _HAS_OPENAI = True
    except Exception:
        _HAS_OPENAI = False

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
    props = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
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

        pos_test = [
            "pm.test('Status is 2xx', function(){pm.expect(pm.response.code).to.be.within(200,299);});",
            "try { pm.test('Response is valid JSON', function(){ JSON.parse(pm.response.text()); }); } catch(e) { pm.test('JSON parse', function(){pm.expect(false).to.be.true;}); }"
        ]
        neg_test = ["pm.test('Status is 4xx or 5xx for invalid input', function(){pm.expect(pm.response.code).to.be.within(400,599);});"]
        edge_test = ["pm.test('Edge case handled', function(){pm.expect(pm.response.code).to.be.within(200,499);});"]

        if pos:
            coll["item"].append(_build_postman_item(f"Positive - {method} {path}", method, path, body=pos, tests=pos_test))
        if neg:
            coll["item"].append(_build_postman_item(f"Negative - {method} {path}", method, path, body=neg, tests=neg_test))
        if edge:
            coll["item"].append(_build_postman_item(f"Divergence - {method} {path}", method, path, body=edge, tests=edge_test))
    return coll


def _call_llm_generate(swagger: Dict[str, Any], divergences: List[Dict[str, Any]],
                       model: str = "gpt-4o-mini", temperature: float = 0.2,
                       gateway_url: Optional[str] = None, timeout: int = 60):
    """
    Call the LLM (supports both new OpenAI SDK and legacy openai).
    Returns parsed JSON dict (Postman collection) or raises on failure.
    """
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI SDK not installed. Install 'openai' to enable LLM mode.")

    # Large, explicit system prompt based on user instruction
    system_prompt = (
        '''You are an expert QA automation engineer and API testing specialist.

You are given two inputs:
1. A Swagger / OpenAPI specification (JSON or YAML converted to JSON).
2. A divergence report listing mismatches between the Swagger specification and the implementation source code.

Your task:
Generate a valid **Postman Collection (v2.1)** named:
"AI Generated API Divergence Test Suite"

VARIABLES REQUIREMENT (MANDATORY):
The Postman JSON MUST include the following variables in the "variable" section:

[
  { "key": "base_url", "value": "<value extracted from Swagger servers[0].url>" },
  { "key": "api_key", "value": "reqres-free-v1" }
]

These variables MUST exist in the final output.

STRUCTURE RULES:
For EACH divergence issue, generate ONE folder inside "item".
Each folder MUST ALWAYS contain:
  1. Positive Test Case(s)
  2. Negative Test Case(s)
  3. Divergence Test Case(s)
No folder should ever omit any of these categories.

REQUEST GENERATION RULES:
Every request MUST include:
"name" — clear descriptive title
"request" object containing:
    - "method"
    - "url" using "{{base_url}}"
    - "header" containing BOTH of these mandatory entries:

      {
        "key": "Content-Type",
        "value": "application/json"
      },
      {
        "key": "x-api-key",
        "value": "{{api_key}}"
      }

    These two headers MUST be present in **every** test case without exception.

    - "body" (for POST, PUT, PATCH) in raw JSON mode when needed

AUTH & VALUE PROPAGATION RULES:
If a token or ID is returned:
Extract it:
      pm.environment.set("token", pm.response.json().token);
      pm.environment.set("created_id", pm.response.json().id);

Use it in subsequent tests:
      Authorization: Bearer {{token}}
      URL paths using {{created_id}}

TEST SCRIPT RULES:
Each request MUST contain at least one "event" with a "test" script:
Validate HTTP status
Validate presence/absence of fields
Validate schema-like structure where possible
Validate divergence behavior:
    - Missing endpoints → expect 404
    - Extra fields in implementation → assert mismatch
    - Wrong status code according to Swagger → assert mismatch
    - Schema divergence → expect 400 or failed validation
No request may be generated without a test script.

BASE URL RULE:
Extract base URL from Swagger (servers[0].url)
Add it to the Postman collection variables as:
      { "key": "base_url", "value": "<swagger-url>" }
All requests MUST use "{{base_url}}" as the host.

OUTPUT RULES:
Output MUST be strict JSON only.
NO markdown, NO explanation text, NO code fences.
Must fully comply with:
      https://schema.getpostman.com/json/collection/v2.1.0/collection.json
The output must be importable and executable in Postman and Newman without manual changes.

GOAL:
Generate a complete, executable Postman test suite that:
Properly structures all folders and test cases
Validates divergence issues
Includes mandatory API key header for EVERY request
Includes “base_url” and “api_key” variables in the Postman JSON
Includes working Postman test scripts
Supports authentication/token chaining and ID reuse
Works end-to-end in CI/CD using Newman
Requires zero modifications after generation.'''
    )

    payload = {"swagger": swagger, "divergences": divergences}

    if not gateway_url:
        gateway_url = _QBURST_DEFAULT_GATEWAY

    # sanitize env inputs and secrets
    api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "").strip()
    gateway_url = (gateway_url or "").strip()

    # safe masked debug (prints to runner logs; not the full key)
    try:
        masked = (api_key[:6] + "...") if api_key else "<missing>"
        print(f"[ai_testgen] Using gateway: {gateway_url!r}, OPENAI_API_KEY present: {bool(api_key)}, masked prefix: {masked}")
    except Exception:
        pass

    messages_payload = json.dumps(payload)

    # Try new OpenAI client first (OpenAI class)
    # NOTE: different SDKs have different usage. Handle both.
    last_err = None
    try:
        # New SDK path: OpenAI()
        try:
            from openai import OpenAI as OpenAIClient  # type: ignore
            client = OpenAIClient(api_key=api_key, base_url=gateway_url) if api_key else OpenAIClient(base_url=gateway_url)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages_payload}
                ],
                temperature=temperature,
                timeout=timeout
            )
            # extract text content
            text = ""
            try:
                text = resp.choices[0].message.content
            except Exception:
                text = getattr(resp.choices[0], "text", "")
            return _safe_json_load(text)
        except Exception as e_new:
            last_err = e_new
            # Fall through to legacy openai path
            pass

        # Legacy openai library path
        try:
            import openai as _openai_legacy  # type: ignore
            if gateway_url:
                _openai_legacy.api_base = gateway_url
            _openai_legacy.api_key = api_key
            resp = _openai_legacy.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages_payload}
                ],
                temperature=temperature
            )
            text = ""
            try:
                # new style
                text = resp.choices[0].message['content']
            except Exception:
                # fallback older format
                text = getattr(resp.choices[0], "text", "")
            return _safe_json_load(text)
        except Exception as e_legacy:
            last_err = e_legacy
            raise RuntimeError(f"LLM generation failed (both SDK attempts). new_err={last_err} legacy_err={e_legacy}")
    except Exception as final_e:
        raise RuntimeError(f"LLM generation failed: {final_e}")


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
