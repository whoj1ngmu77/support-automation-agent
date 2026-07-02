import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")

DEPARTMENT_DOCS = {
    "Sales": ["pricing_guide.txt", "faq.txt"],
    "Technical": ["technical_manual.txt", "faq.txt"],
    "Billing": ["company_policy.txt", "faq.txt"],
    "Account": ["faq.txt", "company_policy.txt"],
}


def _chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    chunks = []
    for s in sections:
        if len(s) <= chunk_size:
            chunks.append(s)
        else:
            for i in range(0, len(s), chunk_size):
                chunks.append(s[i:i + chunk_size])
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.sources = []
        for fname in os.listdir(KB_DIR):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(KB_DIR, fname), encoding="utf-8") as f:
                text = f.read()
            for chunk in _chunk_text(text):
                self.chunks.append(chunk)
                self.sources.append(fname)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, department: str = None, k: int = 3) -> list[str]:
        allowed_files = DEPARTMENT_DOCS.get(department) if department else None

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = []
        for idx in ranked:
            if allowed_files and self.sources[idx] not in allowed_files:
                continue
            if sims[idx] <= 0:
                continue
            results.append(self.chunks[idx])
            if len(results) >= k:
                break
        return results
