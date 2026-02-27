from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple


@dataclass
class NLPResult:
    ok: bool
    detail: str
    data: Dict[str, Any]


def _simple_retrieve_context(text: str, query: str, k: int = 3) -> str:
    """
    Tiny 'RAG-like' retrieval: returns top-k paragraphs with most keyword overlap.
    Works offline (no downloads), so your demo never dies.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    q_tokens = set(query.lower().split())
    scored: List[Tuple[int, str]] = []
    for p in paras:
        p_tokens = set(p.lower().split())
        scored.append((len(q_tokens & p_tokens), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored[:k] if s > 0]
    return "\n\n".join(top) if top else (paras[0] if paras else "")


def classify_text(text: str) -> NLPResult:
    """
    NLP Task 1: Text classification (sentiment-style).
    Uses HuggingFace pipeline if available, otherwise fallback.
    """
    text = (text or "").strip()
    if not text:
        return NLPResult(ok=False, detail="Empty text", data={})

    try:
        from transformers import pipeline  # heavy import inside function

        # Use a known working sentiment model.
        # If downloads fail on slow internet, fallback below still works.
        clf = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        out = clf(text[:512])[0]
        return NLPResult(ok=True, detail="hf", data=out)
    except Exception as e:
        # Offline fallback: rule-based
        pos_words = {"good", "great", "clean", "improved", "safe", "excellent", "love"}
        neg_words = {"bad", "poor", "dirty", "polluted", "unsafe", "hate", "worse"}
        t = text.lower()
        score = sum(w in t for w in pos_words) - sum(w in t for w in neg_words)
        label = "POSITIVE" if score >= 0 else "NEGATIVE"
        return NLPResult(ok=True, detail=f"fallback ({type(e).__name__})", data={"label": label, "score": float(score)})


def qa_over_readme(question: str, project_root: Path | None = None) -> NLPResult:
    """
    NLP Task 2: Question Answering over your README (RAG-lite).
    Tries HF QA pipeline; if it cannot load/download, uses retrieval + extractive fallback.
    """
    question = (question or "").strip()
    if not question:
        return NLPResult(ok=False, detail="Empty question", data={})

    root = project_root or Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    context = ""
    if readme.exists():
        context = readme.read_text(encoding="utf-8", errors="ignore")
    else:
        context = "README.md not found. Add a README with setup + usage + architecture."

    # Keep context bounded
    retrieved = _simple_retrieve_context(context, question, k=4)
    retrieved = retrieved[:4000]

    try:
        from transformers import pipeline

        qa = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
        )
        out = qa(question=question[:256], context=retrieved)
        return NLPResult(ok=True, detail="hf", data=out)
    except Exception as e:
        # Offline fallback: return retrieved context snippet as "answer"
        return NLPResult(
            ok=True,
            detail=f"fallback ({type(e).__name__})",
            data={"answer": retrieved[:800], "score": 0.0, "note": "Fallback retrieval answer"},
        )