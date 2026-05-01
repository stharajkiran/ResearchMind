import os
from anthropic import Anthropic
from dotenv import load_dotenv
import instructor
from researchmind.ingestion.models import Chunk

# Load environment variables from .env file
load_dotenv()

# Patch your client


def build_prompt(query: str, retrieved_chunks: list[Chunk]) -> tuple[str, str]:
    # Create a concise string of ID + Abstract for the prompt

    SYSTEM_PROMPT = f"""
                You are an expert Research Intelligence Agent for ResearchMind. 
                Your core objective is to synthesize technical answers based EXCLUSIVELY on the provided research paper chunks.

                Operational Guidelines:
                1. Grounding: If the answer is not contained within the provided context, state "I do not have sufficient information in the retrieved context to answer this." Do not attempt to use outside knowledge.
                2. Attribution: You must reference specific findings or methods from the text. 
                3. Logic: For comparative or multi-hop queries, explicitly link the findings from different chunks.
                4. Tone: Academic, concise, and technically rigorous. 

                Instructions for Synthesis:
                - Synthesize across all provided chunks. 
                - If chunks contradict each other, explicitly note the discrepancy.
                - Ensure all technical nomenclature (model names, datasets, math notation) is preserved exactly as it appears in the source.
            """

    # context = f"section: {chunk.section}\nText: {chunk.text}"
    chunk_texts = "\n\n".join(
        [
            f"Paper Id: {c.paper_id} | Title: {c.title} | Section: {c.section} | Content: {c.text}"
            for c in retrieved_chunks
        ]
    )
    context = f"RELEVANT RESEARCH CHUNKS:\n---\n{chunk_texts}\n---\nSynthesize the answer based on the chunks above."

    content = f"Context: {context}\n\nQuery: {query}"
    return SYSTEM_PROMPT, content


def build_comparison_prompt(
    query: str, compared_chunks: dict[str, list[Chunk]]
) -> tuple[str, str]:

    SYSTEM_PROMPT = """
        You are an expert Research Intelligence Agent for ResearchMind.
        Your task is to produce a structured, evidence-grounded comparison across multiple methods or approaches.

        Operational Guidelines:
        1. Per-method analysis: Summarise each method independently based ONLY on its provided chunks. Do not blend information across methods.
        2. Grounding: Every claim must be traceable to a specific chunk. If a method's chunks do not contain enough information, state that explicitly for that method.
        3. Comparison: After summarising each method, compare them directly — cover key differences, tradeoffs, performance on datasets, and which performs better and under what conditions.
        4. Attribution: Reference paper IDs for every claim.
        5. Tone: Academic, concise, and technically rigorous.
        6. Preserve all technical nomenclature exactly as it appears in the source chunks.
    """

    sections = []
    for subject, chunks in compared_chunks.items():
        chunk_texts = "\n\n".join(
            [
                f"Paper ID: {c.paper_id} | Title: {c.title} | Section: {c.section} | Content: {c.text}"
                for c in chunks
            ]
        )
        sections.append(f"## {subject}\n{chunk_texts}")

    context = "\n\n".join(sections)
    content = f"RESEARCH CHUNKS BY METHOD:\n---\n{context}\n---\n\nQuery: {query}\n\nProduce a structured comparison across all methods above."

    return SYSTEM_PROMPT, content


def build_gap_prompt(query: str, chunks: list[Chunk]) -> str:
    ## separate limitations/conclusion chunks from the rest
    # priority_chunks = [c for c in chunks if c.section in ("limitations", "conclusion")]
    # other_chunks = [c for c in chunks if c.section not in ("limitations", "conclusion")]
    # ordered_chunks = priority_chunks + other_chunks

    GAP_DETECTION_PROMPT = """You are an expert Research Intelligence Agent specialising in systematic literature analysis.

        Your task is to identify genuine research gaps from the provided paper chunks on a given topic.

        A research gap is one of the following:
        - A problem explicitly mentioned as unsolved or as future work by the authors
        - A missing benchmark or evaluation that multiple papers acknowledge
        - An unexplored combination of methods that the literature implies but has not attempted
        - Contradictory findings across papers that have not been reconciled

        Operational Guidelines:
        1. Grounding: Every gap must be supported by evidence in the provided chunks. Reference the exact paper IDs that support each gap.
        2. Do not invent gaps that are not implied by the text. If the chunks do not contain enough evidence for a gap, do not include it.
        3. Specificity: Vague gaps like "more research is needed" are not acceptable. Every gap must describe a concrete unsolved problem.
        4. Confidence: Assign higher confidence to gaps mentioned explicitly by multiple papers. Assign lower confidence to gaps implied but not stated.
        5. Preserve all technical nomenclature exactly as it appears in the source chunks.

        Examples of strong gaps:
        - "No method has benchmarked prototype-based OOD detection on datasets beyond MVTec and VisA — generalisation to industrial domains is unvalidated."
        - "DINO and DINOv2 features are used for anomaly detection but no work has compared their representations systematically on the same benchmark."

        Examples of weak gaps to avoid:
        - "Further research is needed in this area."
        - "More datasets should be explored."
        """

    chunk_texts = "\n\n".join(
        [
            f"Paper ID: {c.paper_id} | Section: {c.section} | Content: {c.text}"
            for c in chunks
        ]
    )
    content = (
        f"RESEARCH TOPIC: {query}\n\n"
        f"RETRIEVED PAPER CHUNKS:\n---\n{chunk_texts}\n---\n\n"
        f"Identify all research gaps supported by the chunks above."
    )

    return GAP_DETECTION_PROMPT, content
