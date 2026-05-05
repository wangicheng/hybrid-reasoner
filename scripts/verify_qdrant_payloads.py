"""Diagnostic script: verify Qdrant payload format for metadata pre-filtering."""
import sys
sys.path.insert(0, ".")

from src.core.vector_store import VectorStore

vs = VectorStore(collection_name="novels")

# Check collection info
collection_info = vs.client.get_collection(collection_name="novels")
print(f"Collection: novels")
print(f"  Points count: {collection_info.points_count}")
print(f"  Payload indexes: {collection_info.payload_schema}")
print()

# Sample 5 payloads
points, _ = vs.client.scroll(collection_name="novels", limit=5, with_payload=True, with_vectors=False)
for p in points:
    payload = p.payload or {}
    print(f"ID: {p.id}")
    ps = payload.get("publish_status")
    print(f"  publish_status: type={type(ps).__name__}, value={ps!r}")
    wt = payload.get("words_total")
    print(f"  words_total:    type={type(wt).__name__}, value={wt!r}")
    tags = payload.get("tags")
    print(f"  tags:           type={type(tags).__name__}, value={str(tags)[:120]}")
    author = payload.get("author")
    print(f"  author:         type={type(author).__name__}, value={author!r}")
    print()

vs.client.close()
