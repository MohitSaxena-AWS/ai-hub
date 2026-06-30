"""Privacy-preserving business key.

Duplicate detection (assignment section 2.3) must work *without* exposing
personal information. The approach: derive a stable hash from the non-PII
"business" fields only (e.g. request_type + environment + justification). The
hash never contains the requester's name, employee ID or contact details, so it
can be stored and compared freely.

This module owns the deterministic part of the key (an exact-match hash). The
fuzzy/semantic comparison is layered on top in the duplicate service (stage 5).
"""

from __future__ import annotations

import hashlib
from typing import Any


def normalize_value(value: Any) -> str:
    """Normalise a field value for stable hashing.

    Lower-cases and collapses internal whitespace so that cosmetically
    different but semantically identical inputs hash the same.
    """

    return " ".join(str(value).strip().lower().split())


def compute_business_hash(collected: dict[str, Any], business_field_names: list[str]) -> str:
    """SHA-256 over the normalised business fields, in a fixed order.

    Only the named business fields are included; PII fields are deliberately
    excluded so the hash carries no personal information.
    """

    parts = [f"{name}={normalize_value(collected.get(name, ''))}" for name in business_field_names]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
