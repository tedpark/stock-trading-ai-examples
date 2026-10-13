#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements-service.txt
python -m uvicorn paper_service.app:app --host 127.0.0.1 --port 8766
