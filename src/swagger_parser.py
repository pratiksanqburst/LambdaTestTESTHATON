import yaml, json

def parse_swagger(swagger_path: str):
    with open(swagger_path, 'r', encoding='utf-8') as f:
        if swagger_path.endswith('.json'):
            spec = json.load(f)
        else:
            spec = yaml.safe_load(f)

    paths = spec.get('paths', {})
    endpoints = {}

    for path, methods in paths.items():
        for method, details in methods.items():
            summary = details.get('summary', '')
            operation_id = details.get('operationId', f"{method}_{path}")
            responses = list(details.get('responses', {}).keys())
            schema = None
            if 'requestBody' in details:
                content = details['requestBody'].get('content', {})
                for _, v in content.items():
                    schema = v.get('schema', {})
                    break

            endpoints[f"{method.upper()} {path}"] = {
                "method": method.upper(),
                "path": path,
                "summary": summary,
                "operationId": operation_id,
                "responses": responses,
                "schema": schema
            }
    return endpoints
