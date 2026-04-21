from locust import HttpUser, task, between
import json
import random

from researchmind.utils.find_root import find_project_root


project_root = find_project_root()


class ResearchUser(HttpUser):
    wait_time = between(1, 3)  # seconds between tasks per user
    queries_path = project_root / "experiments" / "60_query_test_set.json"
    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)        

    @task
    def search(self):
        # pick a random query from your test set
        query = random.choice(self.queries)["query"]
        # POST to /search
        self.client.post("/search", json={"query": query, "k": 5})
