import logging

import mlflow
import numpy as np
from sklearn.cluster import KMeans

from researchmind.embedding.models import BaseResearchEncoder, MPNetEncoder
from researchmind.feedback.interfaces import FeedbackStore
from researchmind.utils.llm_client import ResearchMindLLM

logger = logging.getLogger(__name__)


class FeedbackLoop:
    def __init__(
        self,
        store: FeedbackStore,
        encoder: BaseResearchEncoder,
        llm: ResearchMindLLM,
    ):
        self.store = store
        self.encoder = encoder
        self.llm = llm

    def run(self, threshold: int = 3) -> None:

        # Load low-rated feedback
        rows = self.store.get_low_rated(threshold=threshold)
        if not rows:
            logger.info("No low-rated feedback found. Exiting.")
            return
        queries = [r["query"] for r in rows]
        # embed all queries
        embeddings = self.encoder.encode(queries, normalize_embeddings=True)

        if len(rows) < 4:
            logger.warning("Not enough low-rated feedback (%d rows). Need at least 4.", len(rows))
            return

        # k-means clustering
        k = min(max(2, int(np.sqrt(len(queries) / 2))), len(queries))

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        cluster_labels = {}
        for cluster_id in range(k):
            # get queries in this cluster
            cluster_queries = [q for q, l in zip(queries, labels) if l == cluster_id]
            # top three queries as sample for LLM to label the cluster
            sample = cluster_queries[:3]
            prompt = (
                f"These are research queries that got poor ratings. Give a 5-word label describing the topic:\n"
                + "\n".join(sample)
            )
            # get the label from the LLM
            label = self.llm.complete(
                user_prompt=prompt,
                tier="fast",
                max_tokens=20,
            )
            cluster_labels[cluster_id] = label

        # Log clusters and labels to 
        mlflow.set_experiment("feedback_loop")
        with mlflow.start_run(run_name="feedback_loop"):
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("n_clusters", k)
            mlflow.log_metric("low_rated_count", len(rows))

            for cluster_id in range(k):
                cluster_rows = [r for r, l in zip(rows, labels) if l == cluster_id]
                scores = [
                    r["hallucination_score"]
                    for r in cluster_rows
                    if r["hallucination_score"] is not None
                ]
                avg_hallucination = np.mean(scores) if scores else -1.0
                mlflow.log_metric(f"cluster_{cluster_id}_size", len(cluster_rows))
                mlflow.log_metric(
                    f"cluster_{cluster_id}_avg_hallucination", avg_hallucination
                )
            mlflow.log_dict(cluster_labels, "cluster_labels.json")

            logger.info("Logged %d clusters to MLflow.", k)

if __name__ == "__main__":
    from researchmind.feedback.store import PostgresFeedbackStore
    store = PostgresFeedbackStore()
    encoder = MPNetEncoder()
    llm = ResearchMindLLM()
    loop = FeedbackLoop(store, encoder, llm)
    loop.run(threshold=3)