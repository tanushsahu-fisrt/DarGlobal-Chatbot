"""See what the retriever finds for a question, without calling the LLM.

    python scripts/try_retrieval.py "villas for sale in Jeddah under 2 million"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag import Knowledge  # noqa: E402

DEFAULTS = ["Who founded DarGlobal?", "Tell me about Sidra",
            "3 bedroom apartments for rent under 50,000 SAR", "What is Wasalt?"]
kb = Knowledge()
print(f"loaded {len(kb.listings)} listings, {len(kb.chunks)} chunks, data as of {kb.as_of}")
for q in sys.argv[1:] or DEFAULTS:
    r = kb.retrieve(q, [])
    print("\n" + "=" * 78 + f"\nQ: {q}\nfilters: {r['filters']}\n" + "-" * 78)
    print(r["direct"] or r["context"])
