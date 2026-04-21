import json
import os
import time
from anthropic import Anthropic
from typing import List, Dict
import json
import random
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# 1. Setup Client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def generate_queries(batch: List[Dict]) -> str:
    """
    Sends a batch of abstracts to Claude to generate semantic and technical queries.
    """
    # Create a concise string of ID + Abstract for the prompt
    context = "\n\n".join(
        [
            f"ID: {p['paper_id']}\nTitle: {p['title']}\nAuthors: {', '.join(p['authors'][:3])}\nAbstract: {p['abstract']}"
            for p in batch
        ]
    )
    prompt = f"""
    You are an expert AI Research Engineer. I will provide you with a list of paper IDs and abstracts. The query must NOT reuse phrases directly from the abstract.
    The query should be realistic and potentially ambiguous.

    For EACH paper, generate two distinct search queries that a researcher might use to find this paper. 

    1. SEMANTIC query:
    - Describe the problem or methodology without using unique keywords or the title. 
    - DO NOT reuse phrases directly from the abstract
    - - Describe the research problem or goal in plain language
    - Slightly ambiguous — could match multiple papers
    - No model names, author names, or dataset names
    - 10-20 words
    - Difficulty: HARD (requires understanding the concept, not matching keywords)

    2. TECHNICAL query:
    - Include 1-2 specific identifiers: model name, author surname, dataset, or method name
    - Concise: 6-10 words maximum
    - Resembles a real search bar query
    - Difficulty: MEDIUM (keyword match should partially work, semantic should do better)

    OUTPUT FORMAT: A JSON list of objects only. No conversational text.
    [
      {{"query": "...", "relevant_paper_ids": ["ID"], "category": "semantic"}},
      {{"query": "...", "relevant_paper_ids": ["ID"], "category": "technical"}}
    ]

    PAPERS TO PROCESS:
    {context}
    """

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        temperature=0,  # Keep it deterministic
        system="Return ONLY valid JSON. No explanation.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# 3. Execution Logic
def main(batch_size: int = 5, total_queries: int = 60):
    # Load your local papers
    with open("data/processed/papers.jsonl", "r") as f:
        all_papers = [json.loads(line) for line in f]

    # Each paper generates 2 queries (semantic + technical)
    num_papers = total_queries // 2
    # Sample papers to get the desired number of queries
    sample_papers = random.sample(all_papers, num_papers)

    # Process in small batches to stay within token limits
    final_test_set = []

    for i in tqdm(range(0, len(sample_papers), batch_size)):
        batch = sample_papers[i : i + batch_size]
        try:
            # Generate queries for the current batch
            queries = generate_queries(batch)
            # Extract JSON from the response
            text = queries.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]  # drop closing ```
            generated_data = json.loads(text)
            final_test_set.extend(generated_data)
        except Exception as e:
            print(f"Error in batch {i}: {e}")

        time.sleep(1)  # Safety delay

    # Save the result
    with open("scripts/60_query_test_set.json", "w") as f:
        json.dump(final_test_set, f, indent=2)
    print(
        f"Successfully created scripts/60_query_test_set.json with {len(final_test_set)} queries."
    )


if __name__ == "__main__":
    batch_size = 5
    total_queries = 60
    print(f"Generating test set with {total_queries} queries...")
    if bool(os.environ.get("ANTHROPIC_API_KEY")):
        main(batch_size=batch_size, total_queries=total_queries)
    else:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
