import json
import hashlib
from typing import Dict, Any, List


def chromosome_hash(chrom: Dict[str, Any]) -> str:
    # deterministic hash for caching
    # sort keys for stable serialization
    s = json.dumps(chrom, sort_keys=True)
    return hashlib.md5(s.encode("utf8")).hexdigest()


def count_features(mask: List[int]) -> int:
    return int(sum(mask))
