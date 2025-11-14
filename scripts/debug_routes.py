#!/usr/bin/env python3
import os
from src.source_parser import extract_routes_from_project

def main():
    source_dir = os.environ.get("SOURCE_DIR", "backend")
    print(f"Scanning folder: {source_dir}")
    try:
        routes = extract_routes_from_project(source_dir)
    except Exception as e:
        print("Error when extracting routes:", e)
        return

    if not routes:
        print("No routes discovered in folder:", source_dir)
    else:
        print("===== ROUTES DISCOVERED =====")
        for r, info in routes.items():
            print(f"{r}  <-- from file: {info.get('source')}")
if __name__ == '__main__':
    main()
