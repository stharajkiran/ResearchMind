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
