from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_search_returns_results():
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "transformer attention", "k": 5})
        assert response.status_code == 200
        results = response.json()
        assert len(results) > 0
        assert "paper_id" in results[0]
        assert "title" in results[0]
        assert "abstract" in results[0]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200