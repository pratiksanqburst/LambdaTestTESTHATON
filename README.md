# API Contract Divergence CI repository (example)
This repository includes:
- `ci_runner.py` : headless runner that:
  - finds a swagger file
  - extracts routes from source
  - computes divergences
  - generates a Postman collection (LLM if enabled, else heuristic)
  - writes `output/divergence_report.json` and `output/postman_collection.json`
- GitHub Actions workflow `.github/workflows/ci-tests.yml` to run CI on push.
- `src/` : parser, comparator, ai_testgen modules (from your provided code).
- sample swagger and sample Flask source to test locally.

## Quick local run (recommended for testing)
1. Create a Python virtualenv and activate:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   npm install -g newman newman-reporter-html
   ```
2. Run the CI runner locally (heuristic mode):
   ```bash
   python ci_runner.py --output-dir output
   ```
   This will create `output/divergence_report.json` and `output/postman_collection.json`.

3. Run the generated collection with Newman:
   ```bash
   newman run output/postman_collection.json --env-var "base_url=http://localhost:5000" --reporters cli,html,junit --reporter-junit-export newman-reports/junit.xml --reporter-html-export newman-reports/report.html
   ```

## To enable LLM generation in GitHub Actions
- Add repository secrets:
  - `OPENAI_API_KEY` (or `API_KEY`): your LLM key accessible by the ai_testgen.
  - `BASE_URL`: the API host used by the generated tests.
  - `QBURST_GATEWAY` (optional): if using your QBURST gateway.
- The workflow calls `python ci_runner.py --use-llm` and will fall back to heuristic if LLM fails.

## Notes
- The included `src/` modules are the same logic you provided (slightly adapted).
- If your API is not publicly reachable from GitHub Actions, Newman will fail to connect. For CI runs against internal environments, either expose a test URL or use a cloud executor (LambdaTest / HyperExecute) and provide `LT_USERNAME`/`LT_ACCESS_KEY` and add a HyperExecute job.

