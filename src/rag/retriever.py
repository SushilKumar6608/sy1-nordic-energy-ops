import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RAG_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "rag_docs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "saved")

TOP_K_DEFAULT = 3
HIGH_CONFIDENCE = 0.40  # RAG result is reliable
PARTIAL_CONFIDENCE = 0.25  # RAG result exists but is a partial match
# Below PARTIAL_CONFIDENCE: knowledge base does not contain the answer


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever for Nordic Energy Ops knowledge base.

    Supports three retrieval modes:
    1. Standard retrieval: direct FAISS search on raw query embedding
    2. HyDE retrieval: Claude generates a hypothetical ideal document paragraph
       before embedding, dramatically reducing false misses from phrasing mismatch
    3. Confidence-gated retrieval: returns confidence tier alongside results
       so the agent pipeline can decide whether to use RAG, blend with parametric
       knowledge, or fall back entirely to parametric reasoning

    Confidence tiers:
    - HIGH   (score >= 0.40): RAG answer is reliable, cite the source
    - PARTIAL (score 0.25-0.40): RAG answer exists but is a partial match, flag it
    - MISS   (score < 0.25): knowledge base does not contain the answer,
                              trigger parametric fallback
    """

    def __init__(self):
        index_path = os.path.join(MODELS_DIR, "rag_faiss.index")
        chunks_path = os.path.join(RAG_DOCS_DIR, "chunks.json")
        model_path = os.path.join(MODELS_DIR, "rag_embedder_name.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Run build_knowledge_base.py first."
            )

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(model_path, "rb") as f:
            model_name = pickle.load(f)

        self.embedder = SentenceTransformer(model_name)
        self._claude = None  # lazy initialised for HyDE
        print(
            f"RAGRetriever loaded: {self.index.ntotal} chunks, " f"model={model_name}"
        )

    def _get_claude(self):
        """Lazy initialise Anthropic client for HyDE expansion."""
        if self._claude is None:
            import anthropic
            from dotenv import load_dotenv

            load_dotenv()
            self._claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._claude

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text string, normalised."""
        return self.embedder.encode([text], normalize_embeddings=True).astype(
            np.float32
        )

    def _search(
        self, embedding: np.ndarray, k: int, category_filter: str = None
    ) -> list:
        """Run FAISS search and return filtered results."""
        search_k = k * 5 if category_filter else k
        search_k = min(search_k, self.index.ntotal)
        scores, indices = self.index.search(embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if category_filter and chunk["category"] != category_filter:
                continue
            results.append(
                {
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "chunk": chunk["chunk"],
                    "score": float(score),
                    "doc_id": chunk["doc_id"],
                }
            )
            if len(results) >= k:
                break

        return results

    def query(
        self, query_text: str, top_k: int = TOP_K_DEFAULT, category_filter: str = None
    ) -> list:
        """
        Standard retrieval: embed query directly and search FAISS.
        Returns list of result dicts with score, title, chunk.
        """
        embedding = self._embed(query_text)
        return self._search(embedding, top_k, category_filter)

    def query_with_hyde(
        self, query_text: str, top_k: int = TOP_K_DEFAULT, category_filter: str = None
    ) -> tuple:
        """
        HyDE retrieval: Hypothetical Document Embeddings.

        Before searching the vector database, ask Claude to generate a
        hypothetical ideal paragraph that would exist in a district heating
        engineering manual to answer this question. Embed the generated
        paragraph instead of the raw query.

        This bridges the semantic gap between informal operator language
        and formal technical manual language, dramatically reducing false
        misses caused by phrasing mismatch rather than genuine knowledge gaps.

        Returns (results, hyde_paragraph, confidence_tier):
        - results: list of matching chunks
        - hyde_paragraph: the generated hypothetical document (for audit trail)
        - confidence_tier: "HIGH", "PARTIAL", or "MISS"
        """
        client = self._get_claude()

        # Step 1: generate hypothetical document paragraph
        hyde_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system="""You are a technical writer for district heating
and cooling grid engineering manuals. Given an operator question, write a
concise paragraph (3-5 sentences) that would appear in a professional
maintenance manual or standard operating procedure to answer that question.
Use formal technical language, include specific thresholds and measurements
where relevant, and write as if this paragraph already exists in a manual.
Do not say you are generating a hypothetical document — just write the
paragraph directly.""",
            messages=[{"role": "user", "content": query_text}],
        )
        hyde_paragraph = hyde_response.content[0].text.strip()

        # Step 2: embed the hypothetical paragraph instead of the raw query
        hyde_embedding = self._embed(hyde_paragraph)
        results = self._search(hyde_embedding, top_k, category_filter)

        # Step 3: determine confidence tier from top result score
        if not results:
            confidence_tier = "MISS"
        elif results[0]["score"] >= HIGH_CONFIDENCE:
            confidence_tier = "HIGH"
        elif results[0]["score"] >= PARTIAL_CONFIDENCE:
            confidence_tier = "PARTIAL"
        else:
            confidence_tier = "MISS"

        return results, hyde_paragraph, confidence_tier

    def confidence_tier(self, results: list) -> str:
        """Determine confidence tier from standard query results."""
        if not results:
            return "MISS"
        top_score = results[0]["score"]
        if top_score >= HIGH_CONFIDENCE:
            return "HIGH"
        elif top_score >= PARTIAL_CONFIDENCE:
            return "PARTIAL"
        return "MISS"

    def format_context(self, results: list, confidence_tier: str = None) -> str:
        """
        Format retrieved chunks into a context string for LLM injection.
        Optionally includes a confidence annotation for the agent to act on.
        """
        if not results:
            return "No relevant documentation found in knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}: {r['title']} "
                f"(relevance: {r['score']:.3f})]\n{r['chunk']}"
            )

        context = "\n\n---\n\n".join(parts)

        if confidence_tier == "PARTIAL":
            context = (
                "NOTE: Knowledge base match is partial "
                "(similarity 0.25-0.40). Supplement with "
                "engineering principles where documentation "
                "is incomplete.\n\n"
            ) + context
        elif confidence_tier == "MISS":
            context = (
                "NOTE: No relevant documentation found in "
                "knowledge base (similarity < 0.25). The "
                "following response is based on general "
                "district heating engineering principles only. "
                "Do not reference specific grid topology facts "
                "such as node IDs, pipe lengths, or installation "
                "dates.\n\n"
            ) + context

        return context

    def query_and_format(
        self,
        query_text: str,
        top_k: int = TOP_K_DEFAULT,
        category_filter: str = None,
        use_hyde: bool = False,
    ) -> tuple:
        """
        Convenience method: query and return (formatted_context, confidence_tier).
        Set use_hyde=True for HyDE-enhanced retrieval.
        """
        if use_hyde:
            results, hyde_para, tier = self.query_with_hyde(
                query_text, top_k, category_filter
            )
        else:
            results = self.query(query_text, top_k, category_filter)
            tier = self.confidence_tier(results)
            hyde_para = None

        return self.format_context(results, tier), tier, hyde_para


if __name__ == "__main__":
    print("=== RAG Retriever Test (with HyDE) ===\n")
    retriever = RAGRetriever()

    test_cases = [
        # (query, use_hyde, expected_tier)
        ("heating demand spike pipe burst Stockholm", False, "HIGH"),
        ("stuck valve cooling winter Kiruna", False, "HIGH"),
        (
            "what are the thermodynamic consequences of high return line temp",
            True,
            "HIGH",
        ),
        ("bolt torque specification substation bypass valve actuator", True, "MISS"),
        ("why does my heat exchanger have low delta T in January", True, "HIGH"),
    ]

    for query, use_hyde, expected in test_cases:
        print(f"Query: '{query}'")
        print(f"Mode: {'HyDE' if use_hyde else 'Standard'} | Expected tier: {expected}")
        context, tier, hyde_para = retriever.query_and_format(
            query, top_k=2, use_hyde=use_hyde
        )
        print(f"Actual tier: {tier}")
        if hyde_para:
            print(f"HyDE paragraph (first 150 chars): {hyde_para[:150]}...")
        # Show top result score
        if use_hyde:
            results, _, _ = retriever.query_with_hyde(query, top_k=1)
        else:
            results = retriever.query(query, top_k=1)
        if results:
            print(f"Top result: [{results[0]['score']:.3f}] {results[0]['title']}")
        print()
