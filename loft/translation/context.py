from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

@dataclass
class TranslationContext:
    """Preserved context from NL→ASP translation."""
    original_nl: str
    asp_code: str
    predicates_used: List[str]
    key_entities: Dict[str, str]  # entity_id -> original_name
    key_terms: List[str]  # Important legal terms to preserve
    timestamp: datetime = field(default_factory=datetime.now)
    context_id: str = field(default="")

    def __post_init__(self):
        if not self.context_id:
            self.context_id = hashlib.sha256(
                f"{self.original_nl}:{self.asp_code}".encode()
            ).hexdigest()[:12]
