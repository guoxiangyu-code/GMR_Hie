"""
EventInterfaceV1: typed handoff object from Part 2 (P0 Adapter) to Part 3 (HMSA).

This is a **model-internal** strong-typed return object.
It is NOT a dataset input, NOT a set of pre-computed features, and must never be
loaded from disk during formal training / validation / test inference or checkpoint
selection.

Schema
------
event_feat              : B x M x 256   (float32)
event_span              : B x M x 2     (float32, normalized [0,1] start/end)
adapter_event_logit     : B x M         (float32, pre-sigmoid)
adapter_quality_logit   : B x M         (float32, pre-sigmoid)
event_mask              : B x M         (bool, True = valid slot)
query_global            : B x 256       (float32)

Constants
---------
schema_version          = "EventInterfaceV1"
M                       = 10
feature_dim             = 256
span_format             = "normalized_start_end"
mask_direction          = "true_is_valid"

Hashes (set by the caller after checkpoint is fixed)
-----------------------------------------------------
baseline_checkpoint_sha256
public_p0_checkpoint_sha256
feature_manifest_sha256
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch


# ── Fixed schema constants ─────────────────────────────────────────────────────
SCHEMA_VERSION: str = "EventInterfaceV1"
M: int = 10
FEATURE_DIM: int = 256
SPAN_FORMAT: str = "normalized_start_end"
MASK_DIRECTION: str = "true_is_valid"


@dataclasses.dataclass
class EventInterfaceV1:
    """
    Typed handoff from P0 Adapter to HMSA / cardinality head.

    All tensors are on the same device and in float32 (or bool for masks).
    Batch dimension B must be consistent across all fields.
    """

    # ── Required tensors ───────────────────────────────────────────────────────
    event_feat: torch.Tensor            # (B, M, 256)
    event_span: torch.Tensor            # (B, M, 2)  normalized [0,1]
    adapter_event_logit: torch.Tensor   # (B, M)
    adapter_quality_logit: torch.Tensor # (B, M)
    event_mask: torch.Tensor            # (B, M)  bool, True = valid

    # Optional: populated when query_global pooling is available
    query_global: Optional[torch.Tensor] = None  # (B, 256)

    # ── Schema metadata ────────────────────────────────────────────────────────
    schema_version: str = dataclasses.field(default=SCHEMA_VERSION, init=False)
    M: int = dataclasses.field(default=M, init=False)
    feature_dim: int = dataclasses.field(default=FEATURE_DIM, init=False)
    span_format: str = dataclasses.field(default=SPAN_FORMAT, init=False)
    mask_direction: str = dataclasses.field(default=MASK_DIRECTION, init=False)

    # ── Hashes (filled in by downstream code after checkpoint is fixed) ────────
    baseline_checkpoint_sha256: Optional[str] = None
    public_p0_checkpoint_sha256: Optional[str] = None
    feature_manifest_sha256: Optional[str] = None

    # ── Post-init validation ───────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._validate_shapes()
        self._validate_mask_direction()

    def _validate_shapes(self) -> None:
        B = self.event_feat.shape[0]
        assert self.event_feat.shape == (B, M, FEATURE_DIM), (
            f"event_feat shape {self.event_feat.shape} != ({B}, {M}, {FEATURE_DIM})"
        )
        assert self.event_span.shape == (B, M, 2), (
            f"event_span shape {self.event_span.shape} != ({B}, {M}, 2)"
        )
        assert self.adapter_event_logit.shape == (B, M), (
            f"adapter_event_logit shape {self.adapter_event_logit.shape} != ({B}, {M})"
        )
        assert self.adapter_quality_logit.shape == (B, M), (
            f"adapter_quality_logit shape {self.adapter_quality_logit.shape} != ({B}, {M})"
        )
        assert self.event_mask.shape == (B, M), (
            f"event_mask shape {self.event_mask.shape} != ({B}, {M})"
        )
        assert self.event_mask.dtype == torch.bool, (
            f"event_mask dtype {self.event_mask.dtype} != bool"
        )
        if self.query_global is not None:
            assert self.query_global.shape == (B, FEATURE_DIM), (
                f"query_global shape {self.query_global.shape} != ({B}, {FEATURE_DIM})"
            )

    def _validate_mask_direction(self) -> None:
        """Ensure schema constant is consistent (True = valid)."""
        assert self.mask_direction == MASK_DIRECTION, (
            f"mask_direction '{self.mask_direction}' != '{MASK_DIRECTION}'"
        )

    # ── Convenience properties ─────────────────────────────────────────────────
    @property
    def batch_size(self) -> int:
        return self.event_feat.shape[0]

    @property
    def num_valid_per_sample(self) -> torch.Tensor:
        """Returns (B,) int tensor: number of valid event slots per sample."""
        return self.event_mask.sum(dim=1)

    @property
    def mode_score(self) -> torch.Tensor:
        """
        Canonical mode ranking score (pre-selection, not for count decision).
        mode_score_m = sigmoid(event_logit_m) * sigmoid(quality_logit_m)
        Zero-filled for invalid slots.
        """
        score = (
            torch.sigmoid(self.adapter_event_logit)
            * torch.sigmoid(self.adapter_quality_logit)
        )
        return score * self.event_mask.float()

    # ── Schema dict (for JSON serialisation / logging) ─────────────────────────
    def schema_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "M": self.M,
            "feature_dim": self.feature_dim,
            "span_format": self.span_format,
            "mask_direction": self.mask_direction,
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "public_p0_checkpoint_sha256": self.public_p0_checkpoint_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
        }

    # ── Verification (called by Part 3 at start-up) ────────────────────────────
    def verify(
        self,
        expected_b0_sha256: Optional[str] = None,
        expected_p0_sha256: Optional[str] = None,
        expected_fm_sha256: Optional[str] = None,
    ) -> None:
        """
        Validate schema constants, shapes, and (optionally) hashes.
        Raises AssertionError on any mismatch.
        """
        assert self.schema_version == SCHEMA_VERSION
        assert self.M == M
        assert self.feature_dim == FEATURE_DIM
        assert self.span_format == SPAN_FORMAT
        assert self.mask_direction == MASK_DIRECTION
        self._validate_shapes()

        if expected_b0_sha256 is not None:
            assert self.baseline_checkpoint_sha256 == expected_b0_sha256, (
                f"B0 sha256 mismatch: got {self.baseline_checkpoint_sha256}"
            )
        if expected_p0_sha256 is not None:
            assert self.public_p0_checkpoint_sha256 == expected_p0_sha256, (
                f"P0 sha256 mismatch: got {self.public_p0_checkpoint_sha256}"
            )
        if expected_fm_sha256 is not None:
            assert self.feature_manifest_sha256 == expected_fm_sha256, (
                f"feature_manifest sha256 mismatch: got {self.feature_manifest_sha256}"
            )
