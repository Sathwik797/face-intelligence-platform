import os
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import numpy as np

from ml.quality import FaceQualityAssessor, QualityMode, FaceQualityMetrics
from ml.gallery import IdentityGallery
from ml.pipeline import ModernRecognitionPipeline
from app.repositories.base import BaseAttendanceRepository
from app.schemas.identities import EnrolledIdentityInfo, EnrollmentResult


class EnrollmentService:
    """
    Quality-Gated Biometric Identity Enrollment & Gallery Management Service.
    Enforces atomic gallery synchronization across in-memory pipeline, SQLite metadata,
    and serialized disk artifacts under a single concurrency lock.
    """

    def __init__(
        self,
        pipeline: ModernRecognitionPipeline,
        repository: BaseAttendanceRepository,
        assessor: Optional[FaceQualityAssessor] = None,
        gallery_filepath: Optional[str] = None
    ):
        self.pipeline = pipeline
        self.repository = repository
        self.assessor = assessor or FaceQualityAssessor(mode=QualityMode.BALANCED)
        self.gallery_filepath = gallery_filepath
        self._lock = threading.Lock()

        # Synchronize existing gallery identities into repository if not present
        self._sync_existing_gallery()

    def _sync_existing_gallery(self):
        """Ensures identities present in the gallery have SQLite metadata records."""
        with self._lock:
            gallery = self.pipeline.gallery
            for ident in gallery.unique_identities:
                existing = self.repository.get_enrolled_identity(ident)
                count = gallery.get_identity_template_count(ident)
                if existing is None:
                    self.repository.upsert_enrolled_identity(EnrolledIdentityInfo(
                        identity=ident,
                        template_count=count,
                        notes="Imported from initial gallery"
                    ))
                elif existing.template_count != count:
                    existing.template_count = count
                    existing.updated_at = datetime.now(timezone.utc).isoformat()
                    self.repository.upsert_enrolled_identity(existing)

    def enroll_identity(
        self,
        identity: str,
        rgb_image: np.ndarray,
        quality_mode: str = "balanced",
        notes: Optional[str] = None
    ) -> EnrollmentResult:
        """
        Enrolls a new biometric template for an identity after passing FQA checks.
        """
        if not identity or not identity.strip():
            return EnrollmentResult(
                success=False,
                identity="",
                error_code="invalid_identity",
                message="Identity name cannot be empty."
            )
        identity = identity.strip()

        if rgb_image is None or not isinstance(rgb_image, np.ndarray) or rgb_image.size == 0:
            return EnrollmentResult(
                success=False,
                identity=identity,
                error_code="invalid_image",
                message="Invalid or empty image payload."
            )

        # 1. Detect faces using pipeline detector
        faces = self.pipeline.detector.detect_faces(rgb_image)
        if not faces:
            return EnrollmentResult(
                success=False,
                identity=identity,
                error_code="no_face_detected",
                message="No face detected in the enrollment image."
            )

        primary_face = faces[0]

        # 2. Quality-Gate Verification via FaceQualityAssessor
        q_mode = QualityMode(quality_mode.lower()) if quality_mode.lower() in [m.value for m in QualityMode] else QualityMode.BALANCED
        assessor = FaceQualityAssessor(mode=q_mode)
        quality_metrics = assessor.assess(
            image=rgb_image,
            detection=primary_face
        )

        if quality_metrics.quality_status == "poor":
            rejection_str = ", ".join(quality_metrics.rejection_reasons) if quality_metrics.rejection_reasons else "quality thresholds not met"
            return EnrollmentResult(
                success=False,
                identity=identity,
                quality_score=quality_metrics.overall_quality_score,
                quality_status="rejected",
                error_code="quality_rejected",
                message=f"Enrollment image rejected by Face Quality Assessment: {rejection_str}."
            )

        # 3. 5-point alignment
        if primary_face.landmarks is None:
            return EnrollmentResult(
                success=False,
                identity=identity,
                error_code="missing_landmarks",
                message="Facial landmarks could not be localized."
            )

        aligned_crop = self.pipeline.aligner.align(rgb_image, primary_face.landmarks)
        if aligned_crop is None:
            return EnrollmentResult(
                success=False,
                identity=identity,
                error_code="alignment_failed",
                message="Face alignment failed."
            )

        # 4. 512D Embedding Extraction
        embedding = self.pipeline.embedder.embed(aligned_crop)
        if embedding is None:
            return EnrollmentResult(
                success=False,
                identity=identity,
                error_code="embedding_failed",
                message="Deep feature embedding extraction failed."
            )

        # 5. Atomic Gallery & Persistence Synchronization
        with self._lock:
            # a. Add template to in-memory gallery
            self.pipeline.gallery.add_templates(identity=identity, embeddings=embedding)
            template_count = self.pipeline.gallery.get_identity_template_count(identity)

            # b. Atomically save gallery artifact to disk if configured
            if self.gallery_filepath:
                try:
                    self.pipeline.gallery.save(self.gallery_filepath)
                except Exception as e:
                    # Rollback in-memory template addition on disk write failure
                    self.pipeline.gallery.remove_identity(identity)
                    return EnrollmentResult(
                        success=False,
                        identity=identity,
                        error_code="disk_persistence_failed",
                        message=f"Failed to persist gallery archive to disk: {str(e)}"
                    )

            # c. Synchronize SQLite metadata
            now_iso = datetime.now(timezone.utc).isoformat()
            info = EnrolledIdentityInfo(
                identity=identity,
                template_count=template_count,
                created_at=now_iso,
                updated_at=now_iso,
                notes=notes
            )
            self.repository.upsert_enrolled_identity(info)

        return EnrollmentResult(
            success=True,
            identity=identity,
            template_count=template_count,
            quality_score=quality_metrics.overall_quality_score,
            quality_status="accepted",
            message=f"Identity '{identity}' enrolled successfully with {template_count} template(s)."
        )

    def delete_identity(self, identity: str) -> bool:
        """
        Removes an identity from the running gallery, disk artifact, and SQLite metadata.
        """
        if not identity:
            return False

        with self._lock:
            removed_templates = self.pipeline.gallery.remove_identity(identity)
            if self.gallery_filepath and removed_templates > 0:
                try:
                    self.pipeline.gallery.save(self.gallery_filepath)
                except Exception:
                    pass
            db_deleted = self.repository.delete_enrolled_identity(identity)
            return (removed_templates > 0) or db_deleted

    def list_identities(self) -> List[EnrolledIdentityInfo]:
        """Lists all enrolled identities."""
        with self._lock:
            records = self.repository.list_enrolled_identities()
            if not records and self.pipeline.gallery.total_templates > 0:
                self._sync_existing_gallery()
                records = self.repository.list_enrolled_identities()
            return records

    def get_identity(self, identity: str) -> Optional[EnrolledIdentityInfo]:
        """Retrieves metadata for a specific identity."""
        with self._lock:
            return self.repository.get_enrolled_identity(identity)
