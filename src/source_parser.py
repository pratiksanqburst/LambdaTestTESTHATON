import os
import re

def extract_routes_from_project(root_dir):
    """
    Recursively scan a project directory and extract routes from .py and .txt files.
    Returns a dict mapping 'METHOD /path' -> metadata.
    """
    routes = {}
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".py", ".txt")):
                file_path = os.path.join(dirpath, file)
                try:
                    file_routes = extract_routes_from_file(file_path)
                    routes.update(file_routes)
                except Exception:
                    # skip files that can't be parsed/read
                    continue
    return routes


def extract_routes_from_file(filepath):
    """
    Read a file (python source or plain text snippet) and find route decorators:
    - Flask: @app.route("/path"), @bp.route("/path")
    - FastAPI: @app.get("/path"), @router.post("/path")
    Returns dict keyed by 'METHOD /path'
    """
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    routes = {}

    # --- Flask/Blueprint: @app.route('/path', methods=['GET','POST']) or @bp.route(...)
    # capture path and optional methods list
    flask_pattern = re.compile(
        r'@(?:\w+\.)?route\(\s*["\\\'](?P<path>[^"\\\']+)["\\\']\s*(?:,\s*methods\s*=\s*(?P<methods>\[[^\]]+\]))?',
        re.IGNORECASE
    )
    for m in flask_pattern.finditer(code):
        path = m.group("path")
        methods_raw = m.group("methods")
        methods = ["GET"]  # default
        if methods_raw:
            # extract quoted method names from list text
            found = re.findall(r'["\\\'](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["\\\']', methods_raw, re.IGNORECASE)
            if found:
                methods = [x.upper() for x in found]
        for method in methods:
            routes[f"{method} {path}"] = {"framework": "flask", "source": os.path.basename(filepath)}

    # --- FastAPI / Starlette style: @app.get("/path") or @router.post(...)
    fastapi_pattern = re.compile(r'@(?:\w+\.)?(get|post|put|delete|patch|options|head)\(\s*["\\\']([^"\\\']+)["\\\']', re.IGNORECASE)
    for m in fastapi_pattern.finditer(code):
        method = m.group(1).upper()
        path = m.group(2)
        routes[f"{method} {path}"] = {"framework": "fastapi", "source": os.path.basename(filepath)}

    return routes
