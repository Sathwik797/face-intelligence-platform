# Phase 15 Dynamic Identity Enrollment & Gallery Management Subsystem

## Overview
This directory documents the **Dynamic Identity Enrollment & Gallery Management Subsystem** established in Phase 15. It bridges visual face quality evaluation (Phase 8 FQA) and ArcFace 512D deep feature extraction (Phase 4) with atomic multi-template gallery management and SQLite metadata synchronization under a single concurrency lock.

---

## 1. Architecture & Data Flow

$$\text{Enrollment Request (Base64)} \xrightarrow{\text{In-Memory Decode}} \text{YuNet Detection} \xrightarrow{\text{Phase 8 FQA Gate}} \text{ArcFace 512D Extraction} \xrightarrow{\text{Atomic Sync Lock}} \begin{cases} \text{IdentityGallery (In-Memory)} \\ \text{Gallery NPZ Archive (Disk)} \\ \text{SQLite Metadata (Database)} \end{cases}$$

```
Enrollment API / UI (POST /api/v1/identities/enroll)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. EnrollmentService (app/services/enrollment_service.py)   │
│    - Decodes image array purely in memory                   │
│    - Detects primary face & localizes 5 facial landmarks    │
│    - Evaluates visual & geometric quality via FQA           │
│    - Quality Gate: rejects blurry/poor frames (HTTP 422)    │
│    - Generates L2-normalized 512D ArcFace embedding vector  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼ (Single Synchronization Lock)
┌─────────────────────────────────────────────────────────────┐
│ 2. Atomic Synchronization                                   │
│    - Injects embedding template into IdentityGallery        │
│    - Saves compressed NPZ archive (data/embeddings/)        │
│    - Upserts metadata record in SQLite enrolled_identities  │
│    - Instantly accessible to running FaceIntelligenceRuntime│
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. REST API Endpoints (/api/v1/identities/)                 │
│    - GET /api/v1/identities (list all enrolled people)      │
│    - GET /api/v1/identities/<name> (template count, stats)  │
│    - POST /api/v1/identities/enroll (enroll new person)     │
│    - DELETE /api/v1/identities/<name> (delete identity)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Quality-Gated Enrollment Policy
1. **FQA Pre-Verification**: An enrollment frame is assessed against Laplacian blur variance, illumination, contrast, and alignment quality before feature extraction. If the face fails quality requirements, the API rejects the request with HTTP 422 and a descriptive error message without corrupting the gallery.
2. **Atomic Synchronization**: In-memory `IdentityGallery`, disk archive (`arcface_gallery.npz`), and SQLite table (`enrolled_identities`) are updated under a single `threading.Lock`. If disk persistence fails, in-memory modifications are rolled back.
3. **Hot-Reloading**: The running `FaceIntelligenceRuntime` recognizes newly enrolled identities on the very next camera frame without process restart or session interruption.

---

## 3. Database Schema (SQLite: `enrolled_identities`)
| Column | Type | Constraints | Description |
|---|---|---|---|
| `identity` | `TEXT` | `PRIMARY KEY` | Enrolled individual name |
| `template_count` | `INTEGER` | `NOT NULL DEFAULT 1` | Total enrolled reference templates |
| `created_at` | `TEXT` | `NOT NULL` | ISO-8601 UTC creation timestamp |
| `updated_at` | `TEXT` | `NOT NULL` | ISO-8601 UTC update timestamp |
| `notes` | `TEXT` | `NULLABLE` | Administrative or enrollment notes |
