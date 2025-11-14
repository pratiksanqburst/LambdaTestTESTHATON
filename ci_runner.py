#!/usr/bin/env python3
"""CI runner to generate divergence report + Postman collection (LLM or heuristic),
then optionally run collection with newman (or upload to HyperExecute).
Usage examples:
  python ci_runner.py --output-dir output
  python ci_runner.py --use-llm --gateway-url "https://llmgateway.qburst.build/v1" --output-dir output
"""
import os
import json
import argparse
from pathlib import Path

# import your modules (adjust import paths if needed)
from src.swagger_parser import parse_swagger
from src.source_parser import extract_routes_from_project
from src.comparator import compare
from src.ai_testgen import generate_postman_collection

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"Wrote {path}")

def find_swagger(root: Path, provided: str = None):
    if provided:
        p = Path(provided)
        if p.exists():
            return str(p)
    # search for common swagger file names
    for name in ["openapi.yaml", "openapi.yml", "swagger.yaml", "swagger.yml", "openapi.json", "swagger.json", "sample_swagger.yaml"]:
        cand = root / name
        if cand.exists():
            return str(cand)
    # fallback: try first .yaml/.yml/.json in repo root
    for ext in ("*.yaml", "*.yml", "*.json"):
        for p in root.glob(ext):
            return str(p)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=".", help="Root of code to scan (default: repo root)")
    parser.add_argument("--swagger-path", default=None, help="Path to swagger file (optional)")
    parser.add_argument("--output-dir", default="output", help="Where to write outputs")
    parser.add_argument("--use-llm", action="store_true", help="Call LLM gateway (requires OPENAI_API_KEY / API_KEY env)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name if use_llm")
    parser.add_argument("--gateway-url", default=None, help="LLM gateway URL (optional)")
    args = parser.parse_args()

    repo_root = Path('.').resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    swagger_file = find_swagger(repo_root, args.swagger_path)
    if not swagger_file:
        print("No swagger file found automatically. Exiting with code 2.")
        raise SystemExit(2)

    # parse swagger endpoints using parse_swagger (returns endpoints map)
    try:
        swagger_endpoints = parse_swagger(swagger_file)
    except Exception as e:
        print("Warning: parse_swagger failed:", e)
        swagger_endpoints = {}

    # attempt to also load the full swagger dict for LLM input
    swagger_dict_full = {}
    try:
        import json, yaml
        with open(swagger_file, 'r', encoding='utf-8') as f:
            text = f.read()
            try:
                swagger_dict_full = json.loads(text)
            except Exception:
                try:
                    swagger_dict_full = yaml.safe_load(text)
                except Exception:
                    swagger_dict_full = {}
    except Exception:
        swagger_dict_full = {}

    # extract code routes
    try:
        code_routes = extract_routes_from_project(args.source_dir)
    except Exception as e:
        print("extract_routes_from_project failed:", e)
        code_routes = {}

    # compare -> divergences
    try:
        divergences = compare(swagger_endpoints, code_routes)
    except Exception as e:
        print("compare failed:", e)
        divergences = []

    # write divergence report
    write_json(out_dir / "divergence_report.json", {"divergences": divergences})

    # generate postman collection (call LLM if requested)
    try:
        use_llm = bool(args.use_llm)
        collection = generate_postman_collection(swagger=swagger_dict_full or {}, divergences=divergences,
                                                use_llm=use_llm, model=args.model, gateway_url=args.gateway_url)
        if not isinstance(collection, dict):
            raise RuntimeError("generate_postman_collection returned non-dict")
    except Exception as e:
        print("Postman generation (LLM) failed - falling back to heuristic. Error:", e)
        collection = generate_postman_collection(swagger=swagger_dict_full or {}, divergences=divergences, use_llm=False)

    write_json(out_dir / "postman_collection.json", collection)

    print("CI runner complete. Outputs:")
    print(" -", out_dir / "divergence_report.json")
    print(" -", out_dir / "postman_collection.json")

if __name__ == "__main__":
    main()
