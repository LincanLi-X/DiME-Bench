from __future__ import annotations

import re


def extract_python_completion(prompt: str, output: str) -> str:
    text = output.strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced[0].strip()
    # If model repeated the prompt, keep only the new suffix.
    if text.startswith(prompt):
        text = text[len(prompt) :]
    # Stop at common chatty tails.
    stops = ["\n\n#", "\n\nif __name__", "\n\nExplanation:", "\n\nThe code"]
    for stop in stops:
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.rstrip() + "\n"
