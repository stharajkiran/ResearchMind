import json
from pathlib import Path
import random
import os
import instructor
from pydantic import BaseModel, Field
from anthropic import Anthropic
import logging
from researchmind.ingestion.models import Chunk
from researchmind.utils.find_root import find_project_root
from researchmind.utils.logging import configure_logging
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CATEGORIES = ["Factual", "Comparative", "Multi-hop", "Temporal", "Gap Detection"]
logger = logging.getLogger(__name__)


class TestQuery(BaseModel):
    question: str  # the query text
    ground_truth: str  # expected answer
    reference_contexts: list[str]  # list of relevant chunk texts that should be retrieved (ground truth for retrieval).
    category: str  # the category of the query
    chunk_id: str  # the ID of the chunk


# Define your strict schema
class GeneratedQuery(BaseModel):
    question: str = Field(..., description="Question that this chunk answers")
    ground_truth: str = Field(..., description="Answer based only on the chunk text")


# Patch your client
client = instructor.from_anthropic(
    Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
)


def generate_queries(chunk: Chunk, category: str) -> TestQuery:
    # Create a concise string of ID + Abstract for the prompt
    context = f"section: {chunk.section}\nText: {chunk.text}"
    prompt = f"""
                You are a Synthetic Data Engineer building an evaluation dataset for an RAG-based Research Intelligence Platform.

                Generate a {category} question that the following research paper chunk answers.
    
                The question must fit this category: {category}
                - Factual: a concept or technique that appears in the text 
                (e.g. "How does X work?" not "What does this paper claim about X?")
                - Comparative: compare two named methods or approaches mentioned in the text
                - Multi-hop: requires connecting a concept in this chunk to a broader research area
                - Temporal: about recent advances in the topic area, not about this paper specifically
                - Gap Detection: what remains unsolved in this research area based on limitations mentioned

                Constraints:
                1. Category : You MUST generate a question that fits the input category.
                2. Grounding: The 'ground_truth' answer MUST be derived ONLY from the provided text. If the text does not contain enough info, return null.
                3. Structure: Output ONLY valid JSON. No markdown blocks, no conversational text.
                4. If the chunk does not contain enough information to generate a {category} question, generate a Factual question instead.
                5. Generalization: The question must NOT reference "this paper", "this work", 
                    "this method", "the authors". It must be askable by a researcher 
                    who has never seen this paper.

                BAD: "What is the main contribution of this work regarding EBL algorithms?"
                GOOD: "How do energy-based learning algorithms handle increasing task difficulty?"

                Then generate a concise answer to the question based solely on the chunk text. Answer in 1-3 sentences maximum

                INPUT DATA:
                
                Chunk:
                Section: {chunk.section}
                Text: {chunk.text}

                SCHEMA:
                {{
                "question": "string (6-10 words, technical/specific)",
                "ground_truth": "string (direct, concise answer based on input)",
                }}
            """
    generated_query: GeneratedQuery = client.chat.completions.create(
        model="claude-haiku-4-5-20251001",
        response_model=GeneratedQuery,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,  # Keep it deterministic
    )

    return TestQuery(
        question=generated_query.question,
        ground_truth=generated_query.ground_truth,
        reference_contexts=[chunk.text],
        category=category,
        chunk_id=chunk.chunk_id,
    )


def sample(chunks: list[Chunk], n_samples: int) -> list[Chunk]:
    """
    Given the sections, we want to sample a diverse set of chunks with weighted distribution.
    """
    HIGH_WEIGHT = {"method", "experiment", "result", "ablationstudy"}
    MEDIUM_WEIGHT = {
        "introduction",
        "background",
        "conclusion",
        "discussion",
        "futurework",
        "limitation",
    }
    LOW_WEIGHT = {"abstract", "relatedwork", "reference", "fulltext", "approach"}

    SECTION_WEIGHTS = {
        **{s: 3 for s in HIGH_WEIGHT},
        **{s: 2 for s in MEDIUM_WEIGHT},
        **{s: 1 for s in LOW_WEIGHT},
    }
    sampled_chunks = random.choices(
        chunks, weights=[SECTION_WEIGHTS.get(c.section, 1) for c in chunks], k=n_samples
    )
    return sampled_chunks


def generate_test_set(
    chunks: list[Chunk], samples_per_category: int = 40
) -> list[TestQuery]:
    """Generate a test set of queries from a list of chunks.

    Args:
        chunks (list[Chunk]): list of chunks which will be sampled to generate queries from

    Returns:
        list[TestQuery]: list of generated test queries
    """

    test_queries = []
    total_queries = len(CATEGORIES) * samples_per_category

    logger.info(
        f"Generating test queries for sampled chunks across {len(CATEGORIES)} categories."
    )
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task(
            "[cyan]Total query generation", total=total_queries
        )
        for category in CATEGORIES:
            # Each category contributes a fixed number of generated queries.
            sampled_chunk = sample(chunks, n_samples=samples_per_category)
            logger.info(
                f"Sampled {len(sampled_chunk)} chunks for category '{category}'."
            )
            category_task = progress.add_task(
                f"[green]{category}", total=len(sampled_chunk)
            )
            for chunk in sampled_chunk:
                test_queries.append(generate_queries(chunk, category))
                progress.update(category_task, advance=1)
                progress.update(overall_task, advance=1)
            progress.remove_task(category_task)
    return test_queries


def main():
    # Load chunks
    with open("data/processed/cleaned_chunks.jsonl", "r") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]

    logger.info(f"Loaded {len(chunks)} chunks for test set generation.")
    # Generate test queries from the sampled chunks
    test_set = generate_test_set(chunks)

    logger.info(
        f"Generated {len(test_set)} test queries across {len(CATEGORIES)} categories."
    )
    # Save the test set to a JSON file
    with open("data/processed/200_test_queries_set.json", "w") as f:
        json.dump([t.model_dump(mode="json") for t in test_set], f, indent=2)
    logger.info(f"Generated test set with {len(test_set)} queries.")


if __name__ == "__main__":
    project_root = find_project_root()
    experiment_name = "200_query_test_set_generation"
    logs_dir = project_root / "logs" / experiment_name
    logfile = logs_dir / "test_set_generation.log"

    configure_logging(logfile, logger)
    main()
    logger.info("Test set generation completed successfully.")
