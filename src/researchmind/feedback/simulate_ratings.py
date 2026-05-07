from dotenv import load_dotenv
load_dotenv()

from researchmind.feedback.store import FeedbackStore

store = FeedbackStore()
rows = store.get_all()
print(f"Found {len(rows)} feedback rows")

for row in rows:
    score = row["hallucination_score"]
    if score is not None and score < 0.75:
        store.update_rating(row["id"], 2)
    else:
        store.update_rating(row["id"], 4)

print("Ratings assigned.")
low = store.get_low_rated(threshold=3)
print(f"Low rated rows: {len(low)}")
