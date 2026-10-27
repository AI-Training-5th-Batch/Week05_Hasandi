"""rag_service.py - a small, self-contained RAG maintenance assistant.

This is the shippable service: import RAGService, or run it from the terminal.

    python rag_service.py "How often should lathes be serviced?"
    python rag_service.py              # interactive mode

It retrieves from the handbook in corpus/, refuses out-of-scope questions via a
retrieval-confidence guardrail, and returns the answer with a trace. Everything is
local and CPU-only. The model sits behind generate(), so swapping in a hosted API
later is a one-function change.
"""

import os, sys, glob, time, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
K = 3
TAU = 0.45  # Sets the similarity threshold for the guardrail. If the best match is below 0.45, it refuses to answer
MAX_NEW_TOKENS = 50

PROMPT = (
"""Answer the question using ONLY the context below. If the answer is not 
in the context, reply exactly: I don't know.
Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"""
)


class RAGService:
    def __init__(self, corpus_dir=CORPUS_DIR, k=K, tau=TAU):
        self.k, self.tau = k, tau
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.chunks, self.sources = self._load(corpus_dir)
        self.vecs = self.embedder.encode(
            self.chunks,
            normalize_embeddings=True
        )

    @staticmethod
    def _load(corpus_dir):
        chunks, sources = [], []
        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
            doc = os.path.basename(path)
            for para in [p.strip() for p in open(path).read().split("\n\n") if p.strip()]:
                if len(para.split()) < 4:
                    continue
                chunks.append(para)
                sources.append(doc)
        return chunks, sources

    def _retrieve(self, question):
        q = self.embedder.encode([question], normalize_embeddings=True)[0]
        sims = self.vecs @ q ## calculate how similar the question is to every single paragraph and @ => Dot Product
        top = np.argsort(-sims)[: self.k]
        return [(self.chunks[i], self.sources[i], float(sims[i])) for i in top]

    def _generate(self, prompt):
        ids = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        out = self.llm.generate(**ids, max_new_tokens=MAX_NEW_TOKENS) ## generate => answer in vectors
        return self.tok.decode(out[0], skip_special_tokens=True) ## decode => converting to human readable format (English texts)

    def ask(self, question):
        t0 = time.perf_counter()
        hits = self._retrieve(question)
        top_sim = hits[0][2]

        if top_sim < self.tau:  # guardrail: refuse, don't hallucinate
            return {
                "answer": "I don't know based on the handbook.",
                "refused": True,
                "sources": [],
                "top_sim": round(top_sim, 3),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)
            }

        ctx = "\n".join(c for c, _, _ in hits)
        answer = self._generate(PROMPT.format(ctx=ctx, q=question))

        return {
            "answer": answer,
            "refused": False,
            "sources": sorted({s for _, s, _ in hits}),
            "top_sim": round(top_sim, 3),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)
        }


def main():
    svc = RAGService()

    if len(sys.argv) > 1:
        print(json.dumps(svc.ask(" ".join(sys.argv[1:])), indent=2))
        return

    print("RAG maintenance assistant. Ask a question, or 'quit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break

        if q.lower() in {"quit", "exit", ""}:
            break

        r = svc.ask(q)
        print(f"\n{r['answer']}")
        tag = (
            "refused (out of scope)"
            if r["refused"]
            else f"sources: {', '.join(r['sources'])}"
        )
        print(f"[{tag}, top_sim={r['top_sim']}  {r['latency_ms']} ms]")


if __name__ == "__main__":
    main()