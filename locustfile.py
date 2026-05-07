from locust import HttpUser, task, between


class ResearchMindUser(HttpUser):
    wait_time = between(1, 3) # each user waits 1-3 seconds between requests


    @task
    def query_agent(self):
        self.client.post(
            "/agent",
            json={
                "query": "What are the main approaches to out-of-distribution detection?",
                "session_id": "load_test_session",
            },
        )

    @task
    def search(self):
        self.client.post(
            "/search",
            json={"query": "self-supervised learning vision transformers"},
        )
