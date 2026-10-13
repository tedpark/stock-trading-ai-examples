# Integrated paper-trading service

A runnable educational service connecting **bars → features → training-only HMM → forward inference → risk limits → paper fills → cash/share accounting → SQLite → optional MLflow**. It uses the book's shared `hmm_demo.py` definitions. It does not place broker orders.

## Start locally

From the example repository root, with Python 3.12:

```sh
python -m pip install -r requirements-service.txt
python -m uvicorn paper_service.app:app --host 127.0.0.1 --port 8766
```

Open http://127.0.0.1:8766/docs for the interactive API. No API key or external service is needed for synthetic runs. Keep this unauthenticated learning server bound to localhost.

```sh
curl -X POST http://127.0.0.1:8766/runs \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"first-demo","seed":42,"count":360}'
curl http://127.0.0.1:8766/runs/first-demo
curl http://127.0.0.1:8766/runs
```

The response includes every simulated fill, cash/share/equity timeline, drawdown, costs, training boundary, input hash and tracking status. Use the same request ID and input to retrieve the original result; changing input with that ID returns 409. New experiments need new IDs.

`PAPER_SERVICE_HOME` selects the storage directory (default `paper-data`). SQLite and saved HMM/scaler bundles survive server restarts. `/runs/{id}/predict` loads the saved model and accepts `{ "bars": [...] }` with the same bar format described below. Supply chronological history, including warmup; filtering restarts from the model's initial-state probabilities over that supplied history. The result is a target **fraction of the configured exposure cap**, not a submitted order. Feature columns must match the saved schema.

## Data and execution assumptions

- Synthetic data are the default. Uploaded data use `source="uploaded"` and a `bars` array of `{timestamp, open, close, volume}`. Supply 100–2000 positive finite bars, unique increasing timestamps, at least 60 training feature rows and 20 held-out feature rows. Dates are normalized to UTC. Reject incomplete data instead of silently filling it.
- Training uses only the first `train_fraction` of the completed feature rows. Forward probabilities and the one-step transition matrix produce the next-state raw-return mean. Positive forecasts propose long exposure; otherwise the account holds cash. This simple rule is an educational baseline.
- This is one instrument, long-only, fractional shares, no leverage. `max_exposure` caps the requested weight relative to pre-trade equity. Market movement and costs can change the realized weight. Fees and slippage apply per fill; available cash caps purchases.
- A signal from close t fills at open t+1. Synthetic opens equal the preceding close; uploaded data use their supplied opens. No fill uses its own bar's close signal. The last close liquidates all remaining shares and charges costs.
- Drawdown is observed at the close. Reaching the limit latches a halt, exits at the next open and prevents re-entry. Gaps can exceed the threshold. This is a chronological replay, not an always-on quote stream.
- Taxes, borrowing, dividends, splits and exchange order books are not modeled. For adjusted Yahoo bars, interpret prices as a normalized adjusted-price experiment rather than an exact historical cash account.
- CPU fits and network logging are synchronous requests. Run modest experiments locally. A crash during a claimed run leaves `running` status; inspect it and use a new ID instead of silently submitting duplicate work. Concurrent claims for an ID are atomic. Database/model writes are not a distributed transaction.
- HMM is the integrated strategy. SAC, pair screening, PER and QR-DQN exercises remain separate; this service does not claim to implement the complete architecture of every chapter.

## Optional MLflow HTTP server

In a separate terminal:

```sh
python -m mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///paper-tracking.db \
  --artifacts-destination ./paper-artifacts
```

Then start the service with:

```sh
PAPER_MLFLOW_URI=http://127.0.0.1:5001 \
MLFLOW_HTTP_REQUEST_TIMEOUT=2 MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 \
python -m uvicorn paper_service.app:app --host 127.0.0.1 --port 8766
```

Successful runs record metrics, input hash, complete result JSON and model artifacts. Tracking failure leaves the local run completed with `tracking.status="failed"`; it is not reported as successful remote recording. Restarting the tracking server does not automatically backfill failed uploads. Do not deploy runs merely because paper accounting completed.

## Optional read-only Yahoo data

```sh
python -m pip install -r requirements-market.txt
python -m paper_service.market SPY --output /tmp/spy-paper.json
curl -X POST http://127.0.0.1:8766/runs \
  -H 'Content-Type: application/json' --data-binary @/tmp/spy-paper.json
```

This downloads adjusted daily bars through yfinance and then runs the **same paper engine**. Provider outages, rate limits and revisions are possible; failed downloads raise an error. The command never authenticates to a broker. Edit `request_id` in the JSON before using changed data.

## Verify

```sh
python verify.py
python -m unittest discover -s tests -p test_paper_service.py -v
python verify_service.py --output service-test-data
# Explicitly request a live, read-only provider check:
python verify_service.py --output service-test-data --market-symbol SPY
```

`verify.py` blocks outbound connections and tests all chapter exercises plus the paper service's local logic. `verify_service.py` separately starts real loopback HTTP servers on free ports, verifies MLflow artifact uploads/downloads, duplicate handling, process restart, saved-model inference, uploaded data, a real MLflow outage and the original Chapter 18 monitoring app. It stops its own servers and removes temporary account/model data. A requested live provider failure is recorded separately as `market.status="unavailable"`; it is not a passed provider test.

No IBKR, FMP, Discord, MongoDB, PostgreSQL, Prometheus or Grafana deployment is included or required. Those integrations appear as architectural extensions in the book. SQLite is embedded storage; FastAPI and MLflow run locally in this service.
