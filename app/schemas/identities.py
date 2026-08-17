from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


@dataclass
class EnrolledIdentityInfo:
    """
    Metadata representation of an enrolled person in the biometric gallery.
    """
    identity: str
    template_count: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "template_count": int(self.template_count),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes
        }


@dataclass
class EnrollmentResult:
    """
    Structured outcome of an identity enrollment attempt.
    """
    success: bool
    identity: str
    template_count: int = 0
    quality_score: Optional[float] = None
    quality_status: str = "none"
    message: str = ""
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "identity": self.identity,
            "template_count": int(self.template_count),
            "quality_score": round(float(self.quality_score), 4) if self.quality_score is not None else None,
            "quality_status": self.quality_status,
            "message": self.message,
            "error_code": self.error_code
        }
