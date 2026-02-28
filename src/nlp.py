from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class NLPResult:
    ok: bool
    detail: str
    data: Dict[str, Any]


# -----------------------------
# 1) Text classification (sentiment) - VADER (offline + lightweight)
# -----------------------------
_analyzer = SentimentIntensityAnalyzer()


def classify_text(text: str) -> NLPResult:
    text = (text or "").strip()
    if not text:
        return NLPResult(ok=False, detail="Empty text", data={})

    scores = _analyzer.polarity_scores(text)
    compound = float(scores.get("compound", 0.0))

    # VADER thresholds
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return NLPResult(
        ok=True,
        detail="vader",
        data={
            "label": label,
            "score": compound,
            "scores": scores,
        },
    )


# -----------------------------
# 2) QA over README (RAG-lite) - TF-IDF retrieval (offline)
# -----------------------------
def _chunk_text(text: str, max_chars: int = 900) -> List[str]:
    # paragraph-based chunking
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def qa_over_readme(question: str, project_root: Path | None = None, top_k: int = 3) -> NLPResult:
    question = (question or "").strip()
    if not question:
        return NLPResult(ok=False, detail="Empty question", data={})

    root = project_root or Path(__file__).resolve().parents[1]
    readme_path = root / "README.md"

    if not readme_path.exists():
        return NLPResult(
            ok=True,
            detail="tfidf-rag",
            data={
                "question": question,
                "answer": "README.md not found at project root.",
                "contexts": [],
                "readme_path": str(readme_path),
            },
        )

    readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
    chunks = _chunk_text(readme_text)

    if not chunks:
        return NLPResult(
            ok=True,
            detail="tfidf-rag",
            data={
                "question": question,
                "answer": "README appears empty.",
                "contexts": [],
            },
        )

    # TF-IDF retrieval
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks + [question])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    top_idx = sims.argsort()[::-1][:top_k]

    contexts = [
        {"rank": i + 1, "score": float(sims[idx]), "text": chunks[idx]}
        for i, idx in enumerate(top_idx)
        if sims[idx] > 0
    ]

    answer = contexts[0]["text"] if contexts else "No relevant section found in README."

    return NLPResult(
        ok=True,
        detail="tfidf-rag",
        data={
            "question": question,
            "answer": answer,
            "contexts": contexts,
        },
    )