"""Chapter 18 learning example; dummy data only. Source: 18-monitoring.md"""
from fastapi import FastAPI
from monitor_example import monitor

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/report")
def report():
    return monitor([-200, 200])

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        report = client.get('/report').json()
        assert report['violations'] == ['mdd']
        print(report)
