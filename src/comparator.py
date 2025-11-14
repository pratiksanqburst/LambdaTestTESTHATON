def compare(swagger_endpoints, code_routes):
    divergences = []

    for ep, sw in swagger_endpoints.items():
        method = sw.get('method')
        summary = sw.get('summary', '')
        swagger_responses = set(sw.get('responses', []))
        schema = sw.get('schema', {})

        if ep not in code_routes:
            divergences.append({
                "endpoint": ep,
                "method": method,
                "summary": summary,
                "issue": "Endpoint missing in source code",
                "expected": list(swagger_responses),
                "found": [],
                "risk_score": 0.9
            })
        else:
            found_responses = set(["200", "201"])
            missing_resp = swagger_responses - found_responses
            extra_resp = found_responses - swagger_responses

            if missing_resp:
                divergences.append({
                    "endpoint": ep,
                    "method": method,
                    "summary": summary,
                    "issue": f"Missing responses: {list(missing_resp)}",
                    "expected": list(swagger_responses),
                    "found": list(found_responses),
                    "risk_score": 0.7
                })
            if extra_resp:
                divergences.append({
                    "endpoint": ep,
                    "method": method,
                    "summary": summary,
                    "issue": f"Extra responses: {list(extra_resp)}",
                    "expected": list(swagger_responses),
                    "found": list(found_responses),
                    "risk_score": 0.5
                })

            if schema:
                size = len(str(schema))
                risk = (size % 10) / 10.0
                divergences.append({
                    "endpoint": ep,
                    "method": method,
                    "summary": summary,
                    "issue": "Schema validation needed",
                    "expected": "Valid schema",
                    "found": f"Approx size {size}",
                    "risk_score": round(min(0.8, 0.3 + risk), 2)
                })

    for c_ep in code_routes:
        if c_ep not in swagger_endpoints:
            divergences.append({
                "endpoint": c_ep,
                "method": c_ep.split()[0],
                "summary": "",
                "issue": "Extra route not in Swagger",
                "expected": [],
                "found": [],
                "risk_score": 0.6
            })

    divergences.sort(key=lambda x: x["risk_score"], reverse=True)
    return divergences
