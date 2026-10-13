#!/usr/bin/env bash
set -euo pipefail
curl --fail http://127.0.0.1:8766/health
curl --fail http://127.0.0.1:8766/report
