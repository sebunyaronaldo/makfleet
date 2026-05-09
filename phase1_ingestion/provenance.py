import hashlib
import json


def hash_record(record: dict) -> str:
    """SHA-256 hash of the immutable identity fields of a telemetry record."""
    payload = (
        str(record.get("event_id", ""))
        + str(record.get("vehicle_id", ""))
        + str(record.get("timestamp", ""))
        + str(record.get("lat", ""))
        + str(record.get("lon", ""))
        + str(record.get("ax", ""))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_chain(batch: list[dict]) -> str:
    """
    Merkle-style root hash across a batch of already-hashed records.
    Each record must already contain a 'provenance_hash' field.
    The chain root is stored in Neo4j as a ProvenanceAnchor node so that
    any post-ingestion tampering can be detected by recomputing.
    """
    leaves = [r["provenance_hash"] for r in batch]
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            pair = leaves[i] + (leaves[i + 1] if i + 1 < len(leaves) else leaves[i])
            next_level.append(hashlib.sha256(pair.encode()).hexdigest())
        leaves = next_level
    return leaves[0] if leaves else hashlib.sha256(b"empty").hexdigest()


def stamp_batch(records: list[dict]) -> tuple[list[dict], str]:
    """Add provenance_hash to each record; return records + Merkle root."""
    for r in records:
        r["provenance_hash"] = hash_record(r)
    root = build_chain(records)
    return records, root
