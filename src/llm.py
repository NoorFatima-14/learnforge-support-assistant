import json
import os
import re

from google import genai
from google.genai import types

SYSTEM_INSTRUCTIONS = """
You are the LearnForge customer support assistant. Answer ONLY using the
CONTEXT chunks provided below. Each chunk is labeled with an id like
[faq-02] or [policy-01] or [ticket-08].

Rules:
1. Never invent policy details, prices, dates, or steps that are not in the
   context. If the context doesn't fully answer the question, say so and
   set "grounded": false.
2. Some chunks mention that an older/previous version of a policy said
   something different. Always follow the current rule, not the outdated
   one.
3. If retrieved chunks genuinely conflict and you can't tell which is
   current, say so and lower your confidence rather than guessing.
4. Set "escalate": true for account security issues, fraud/unauthorized
   charges, refund requests outside the documented window, or anything the
   context doesn't clearly answer.
5. Keep the answer concise.

Respond with ONLY a JSON object (no markdown fences) in this shape:
{
  "answer": "<answer to the user>",
  "grounded": true/false,
  "confidence": "high" | "medium" | "low",
  "sources": ["<chunk id>", ...],
  "escalate": true/false,
  "escalation_reason": "<short reason, or null>"
}
""".strip()

REWRITE_INSTRUCTIONS = """
Given the conversation so far and the user's latest message, rewrite the
latest message as a single standalone search query, resolving pronouns or
references to earlier turns. Respond with ONLY the rewritten query text.
""".strip()


def _format_context(retrieved_chunks):
    parts = []
    for rc in retrieved_chunks:
        c = rc.chunk
        parts.append(f"[{c.id}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _format_history(history):
    if not history:
        return "(no prior turns)"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def _parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class GeminiClient:
    def __init__(self, model_name="gemini-3.6-flash", api_key=None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before running.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def rewrite_query(self, history, question):
        if not history:
            return question
        prompt = f"{REWRITE_INSTRUCTIONS}\n\nConversation so far:\n{_format_history(history)}\n\nLatest message: {question}"
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return (resp.text or "").strip() or question

    def generate_answer(self, question, retrieved_chunks, history):
        context = _format_context(retrieved_chunks)
        prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"CONVERSATION HISTORY:\n{_format_history(history)}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n"
        )
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return _parse_json(resp.text)
