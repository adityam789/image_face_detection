"""
facesorter/search/prompts.py
Few-shot prompt for the LLM query extractor.

The model is only asked to *copy* the date phrase from the query, never to compute dates:
asking a 1.5B model for absolute dates scored 43-63%, copying the phrase and resolving it in
code scored 83-100% (see docs/search-design.md). The wording and examples are the ones that
were measured, so change them only together with a re-evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

SYSTEM_PROMPT = (
    "Return ONLY a JSON object with keys: "
    '"text" (what to look for visually; exclude place, people and time words; "" if nothing '
    'visual), "place" (string or null), "people" (list of first names, [] if none), '
    '"date_phrase" (the time expression copied verbatim from the query, or null).'
)

FEW_SHOTS: tuple[tuple[str, Mapping[str, object]], ...] = (
    (
        "pizza night with Tom in Rome in April 2022",
        {"text": "pizza night", "place": "Rome", "people": ["Tom"], "date_phrase": "April 2022"},
    ),
    ("dogs", {"text": "dogs", "place": None, "people": [], "date_phrase": None}),
    (
        "fireworks on the Fourth of July 2021",
        {"text": "fireworks", "place": None, "people": [], "date_phrase": "Fourth of July 2021"},
    ),
    (
        "Anna's wedding photos from autumn 2018 in Kyoto",
        {
            "text": "wedding photos",
            "place": "Kyoto",
            "people": ["Anna"],
            "date_phrase": "autumn 2018",
        },
    ),
    (
        "pictures from last winter",
        {"text": "", "place": None, "people": [], "date_phrase": "last winter"},
    ),
)


def build_messages(query: str) -> list[dict[str, str]]:
    """Chat messages for ``query``: system prompt, the few-shot turns, then the query itself.

    The last user message is exactly ``query`` so scripted test backends can key on it.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_query, answer in FEW_SHOTS:
        messages.append({"role": "user", "content": shot_query})
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    messages.append({"role": "user", "content": query})
    return messages
