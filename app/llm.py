from __future__ import annotations

PROMPTS: dict[str, str] = {
    "tldr": "Summarize the syllabus in 6 bullet points.",
    "workload": "Estimate workload (hours/week), identify heavy weeks, and explain why.",
    "grading": "Extract grading breakdown and major deliverables.",
    "prereqs": "Infer implied prerequisites and recommended background.",
}


def run_llm(prompt_key: str, syllabus_text: str) -> str:
    """Demo placeholder.

    Replace with your actual LLM call.
    If you later add PDF->text extraction, pass the extracted text in.
    """
    prompt = PROMPTS.get(prompt_key, "Unknown prompt")
    return (
        "[DEMO PLACEHOLDER]\n\n"
        f"Prompt key: {prompt_key}\n"
        f"Prompt: {prompt}\n\n"
        "(Later: run LLM on extracted PDF text.)\n"
    )
