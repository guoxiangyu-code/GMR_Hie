"""
tests/test_event_set_metrics.py

Unit tests for Duplicate/FullCoverage/SetSuccess metrics (Part 2 §6).
All toy cases verified to match the document spec exactly.
"""

import unittest
import torch


# ── Metric implementation to test ─────────────────────────────────────────────
def tiou(a: list, b: list) -> float:
    """tIoU of two intervals [start, end]."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / max(union, 1e-9)


def max_bipartite_matching(pred_set: list, gt_set: list, theta: float) -> list:
    """
    Maximum cardinality one-to-one matching on E_theta = {(p,g) | tIoU(p,g) >= theta}.
    Breaks ties by maximising total tIoU.
    Returns list of (pred_idx, gt_idx) matched pairs.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    P, G = len(pred_set), len(gt_set)
    if P == 0 or G == 0:
        return []

    # Build tIoU matrix; entries below theta are 0
    mat = np.zeros((P, G))
    eligible = np.zeros((P, G), dtype=bool)
    for i, p in enumerate(pred_set):
        for j, g in enumerate(gt_set):
            t = tiou(p, g)
            mat[i, j] = t
            eligible[i, j] = (t >= theta)

    # Only match eligible pairs; maximise cardinality then total tIoU
    # Use negative tIoU as cost on eligible subgraph; ineligible pairs cost=+inf
    cost = np.where(eligible, -mat, 1e9)

    # Remove rows/cols with no eligible entry
    row_has = eligible.any(axis=1)
    col_has = eligible.any(axis=0)
    if not row_has.any() or not col_has.any():
        return []

    sub_cost = cost[np.ix_(row_has, col_has)]
    orig_rows = np.where(row_has)[0]
    orig_cols = np.where(col_has)[0]

    r, c = linear_sum_assignment(sub_cost)
    pairs = []
    for ri, ci in zip(r, c):
        pi, gi = orig_rows[ri], orig_cols[ci]
        if eligible[pi, gi]:
            pairs.append((int(pi), int(gi)))
    return pairs


def compute_event_set_metrics(
    predictions: list,   # list of (P_q, G_q) tuples; each is list of [start, end]
    theta: float = 0.5,
) -> dict:
    """
    Compute DuplicateRate, FullCoverage, SetSuccess at theta.
    predictions: [(P_q, G_q), ...] where each P_q and G_q are lists of [s,e].
    """
    total_eligible = 0
    total_duplicate = 0
    fc_queries = []
    ss_list = []

    for P_q, G_q in predictions:
        pairs = max_bipartite_matching(P_q, G_q, theta)
        M_size = len(pairs)

        # Eligible predictions: those with at least one G with tIoU >= theta
        eligible_preds = [
            i for i, p in enumerate(P_q)
            if any(tiou(p, g) >= theta for g in G_q)
        ] if G_q else []

        # Duplicate count: |P_eligible| - |M_theta|
        dup = max(0, len(eligible_preds) - M_size)
        total_eligible += len(eligible_preds)
        total_duplicate += dup

        # FullCoverage: only for |G_q| >= 2
        if len(G_q) >= 2:
            fc_queries.append(1 if M_size == len(G_q) else 0)

        # SetSuccess: |P_q| == |G_q| == |M_theta(q)|
        if len(G_q) == 0 and len(P_q) == 0:
            # null query
            ss_list.append(1)
        elif len(P_q) == len(G_q) == M_size:
            ss_list.append(1)
        else:
            ss_list.append(0)

    dup_rate = total_duplicate / max(total_eligible, 1)
    fc = sum(fc_queries) / max(len(fc_queries), 1) if fc_queries else float("nan")
    ss = sum(ss_list) / max(len(ss_list), 1) if ss_list else float("nan")

    return {
        "DuplicateRate": dup_rate,
        "FullCoverage": fc,
        "SetSuccess": ss,
        "n_fc_queries": len(fc_queries),
        "n_ss_queries": len(ss_list),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEventSetMetrics(unittest.TestCase):

    def test_null_success_both_empty(self):
        """Null query: P_q=[], G_q=[] → SetSuccess=1."""
        metrics = compute_event_set_metrics([([], [])])
        self.assertEqual(metrics["SetSuccess"], 1.0)

    def test_null_failure_pred_nonempty(self):
        """Null query: P_q=[[0,0.5]], G_q=[] → SetSuccess=0."""
        metrics = compute_event_set_metrics([([[0, 0.5]], [])])
        self.assertEqual(metrics["SetSuccess"], 0.0)

    def test_null_failure_gt_nonempty(self):
        """Null query: P_q=[], G_q=[[0,0.5]] → SetSuccess=0."""
        metrics = compute_event_set_metrics([([], [[0, 0.5]])])
        self.assertEqual(metrics["SetSuccess"], 0.0)

    def test_single_gt_perfect_match(self):
        """Single GT, exact match → SetSuccess=1, Duplicate=0."""
        metrics = compute_event_set_metrics([([[0.1, 0.6]], [[0.1, 0.6]])])
        self.assertEqual(metrics["SetSuccess"], 1.0)
        self.assertAlmostEqual(metrics["DuplicateRate"], 0.0)

    def test_single_gt_no_match(self):
        """Single GT, no overlap → SetSuccess=0."""
        metrics = compute_event_set_metrics([([[0.8, 1.0]], [[0.0, 0.2]])])
        self.assertEqual(metrics["SetSuccess"], 0.0)

    def test_single_gt_wrong_count(self):
        """Single GT, 2 preds → SetSuccess=0 (count error)."""
        metrics = compute_event_set_metrics([
            ([[0.1, 0.6], [0.4, 0.9]], [[0.1, 0.6]])
        ])
        self.assertEqual(metrics["SetSuccess"], 0.0)

    def test_multi_gt_full_coverage(self):
        """2 GTs, 2 perfectly matched preds → FullCoverage=1, SetSuccess=1."""
        P_q = [[0.0, 0.4], [0.5, 0.9]]
        G_q = [[0.0, 0.4], [0.5, 0.9]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        self.assertEqual(metrics["FullCoverage"], 1.0)
        self.assertEqual(metrics["SetSuccess"], 1.0)

    def test_multi_gt_partial_coverage(self):
        """2 GTs, only 1 matched → FullCoverage=0."""
        P_q = [[0.0, 0.4]]
        G_q = [[0.0, 0.4], [0.5, 0.9]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        self.assertEqual(metrics["FullCoverage"], 0.0)

    def test_duplicate_detection(self):
        """2 preds both matching same GT → 1 duplicate."""
        # tIoU([0.0,0.5], [0.0,0.5])=1.0 >= 0.5  → eligible
        # tIoU([0.1,0.6], [0.0,0.5])=0.67 >= 0.5 → eligible
        # Only 1 GT → M_theta=1; |eligible|=2; duplicate=1
        P_q = [[0.0, 0.5], [0.1, 0.6]]
        G_q = [[0.0, 0.5]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        self.assertEqual(metrics["DuplicateRate"], 0.5)  # 1/2

    def test_false_positive_not_duplicate(self):
        """A pred that doesn't match any GT is a FP, NOT a duplicate."""
        # [0.8, 1.0] has tIoU<0.5 with GT [0.0,0.5] → not eligible → not duplicate
        P_q = [[0.1, 0.6], [0.8, 1.0]]
        G_q = [[0.0, 0.5]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        # [0.1,0.6] is eligible (tIoU≈0.67>0.5), [0.8,1.0] is not eligible
        # M_theta = 1 (only [0.1,0.6] matched), eligible=1, dup=0
        self.assertAlmostEqual(metrics["DuplicateRate"], 0.0)

    def test_two_preds_two_overlapping_gts_no_duplicate(self):
        """Two preds matching two separate (overlapping) GTs → no duplicate."""
        # Highly overlapping GTs (tIoU=0.857 between them)
        P_q = [[0.0, 0.7], [0.3, 1.0]]
        G_q = [[0.0, 0.7], [0.3, 1.0]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        # Each pred matches its corresponding GT → M_theta=2, eligible=2, dup=0
        self.assertAlmostEqual(metrics["DuplicateRate"], 0.0)
        self.assertEqual(metrics["SetSuccess"], 1.0)

    def test_tiou_threshold_exact(self):
        """tIoU exactly at threshold is included."""
        # tIoU([0,1], [0.5,1]) = 0.5/1.0 = 0.5 → eligible at theta=0.5
        t = tiou([0, 1], [0.5, 1])
        self.assertAlmostEqual(t, 0.5, places=5)
        metrics = compute_event_set_metrics([([[0, 1]], [[0.5, 1]])], theta=0.5)
        self.assertEqual(metrics["SetSuccess"], 1.0)

    def test_set_success_count_mismatch(self):
        """count(P)≠count(G) → SetSuccess=0 even if all GTs covered."""
        P_q = [[0.0, 0.4], [0.5, 0.9], [0.2, 0.7]]
        G_q = [[0.0, 0.4], [0.5, 0.9]]
        metrics = compute_event_set_metrics([(P_q, G_q)])
        self.assertEqual(metrics["SetSuccess"], 0.0)

    def test_mean_over_multiple_queries(self):
        """SetSuccess is macro mean across all queries."""
        # Query 1: success, Query 2: fail
        q1 = ([[0.0, 0.4]], [[0.0, 0.4]])          # success
        q2 = ([[0.0, 0.4]], [[0.6, 1.0]])          # fail (no match)
        metrics = compute_event_set_metrics([q1, q2])
        self.assertAlmostEqual(metrics["SetSuccess"], 0.5)


class TestMaskedMean(unittest.TestCase):

    def test_masked_mean_all_valid(self):
        from models.flash_vtg_gmr.event_cardinality import masked_mean
        x = torch.ones(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = masked_mean(x, mask)
        self.assertEqual(out.shape, (2, 8))
        self.assertAlmostEqual(out.mean().item(), 1.0)

    def test_masked_mean_all_zero_mask(self):
        from models.flash_vtg_gmr.event_cardinality import masked_mean
        x = torch.ones(2, 4, 8)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        counter = [0]
        out = masked_mean(x, mask, audit_counter=counter)
        self.assertEqual(out.shape, (2, 8))
        self.assertAlmostEqual(out.abs().max().item(), 0.0)   # returns zero vector
        self.assertEqual(counter[0], 2)   # 2 rows audited


class TestCountHeadV1(unittest.TestCase):

    def test_isolated_init_identical(self):
        """G0 and C1 CountHeadV1 must have element-wise identical params after isolated init."""
        from models.flash_vtg_gmr.event_cardinality import CountHeadV1, init_count_head_isolated

        # Simulate creating other modules before CountHeadV1 (these should NOT affect it)
        import torch.nn as nn
        _ = nn.Linear(100, 100)
        torch.manual_seed(9999)  # change global RNG

        h1 = CountHeadV1(text_dim=512, set_dim=256)
        init_count_head_isolated(h1, seed=2024, rng_key="CountHeadV1")

        _ = nn.Linear(200, 200)
        torch.manual_seed(12345)

        h2 = CountHeadV1(text_dim=512, set_dim=256)
        init_count_head_isolated(h2, seed=2024, rng_key="CountHeadV1")

        for (n1, p1), (n2, p2) in zip(h1.named_parameters(), h2.named_parameters()):
            self.assertTrue(
                torch.allclose(p1, p2, atol=0.0, rtol=0.0),
                f"Parameter {n1} differs between two isolated-init CountHeadV1"
            )

    def test_different_seeds_differ(self):
        from models.flash_vtg_gmr.event_cardinality import CountHeadV1, init_count_head_isolated
        h1 = CountHeadV1()
        h2 = CountHeadV1()
        init_count_head_isolated(h1, seed=2024)
        init_count_head_isolated(h2, seed=2025)
        # They must differ for at least one parameter
        any_diff = False
        for (n1, p1), (n2, p2) in zip(h1.named_parameters(), h2.named_parameters()):
            if not torch.allclose(p1, p2):
                any_diff = True
                break
        self.assertTrue(any_diff, "Seeds 2024 and 2025 should produce different params")


class TestEventInterfaceV1(unittest.TestCase):

    def _make_iface(self, B=2, M=10, D=256):
        from models.flash_vtg_gmr.event_interface import EventInterfaceV1
        feat = torch.randn(B, M, D)
        span = torch.rand(B, M, 2)
        logit = torch.randn(B, M)
        qual = torch.randn(B, M)
        mask = torch.ones(B, M, dtype=torch.bool)
        qg = torch.randn(B, D)
        return EventInterfaceV1(feat, span, logit, qual, mask, query_global=qg)

    def test_valid_construction(self):
        iface = self._make_iface()
        self.assertEqual(iface.schema_version, "EventInterfaceV1")
        self.assertEqual(iface.M, 10)
        self.assertEqual(iface.mask_direction, "true_is_valid")

    def test_shape_mismatch_raises(self):
        from models.flash_vtg_gmr.event_interface import EventInterfaceV1
        with self.assertRaises(AssertionError):
            EventInterfaceV1(
                event_feat=torch.randn(2, 5, 256),   # wrong M
                event_span=torch.rand(2, 10, 2),
                adapter_event_logit=torch.randn(2, 10),
                adapter_quality_logit=torch.randn(2, 10),
                event_mask=torch.ones(2, 10, dtype=torch.bool),
            )

    def test_mode_score_zero_for_invalid(self):
        iface = self._make_iface()
        # Invalidate last slot
        iface.event_mask[:, -1] = False
        score = iface.mode_score
        self.assertTrue((score[:, -1] == 0.0).all())

    def test_verify_passes(self):
        iface = self._make_iface()
        iface.baseline_checkpoint_sha256 = "abc123"
        iface.verify(expected_b0_sha256="abc123")

    def test_verify_hash_mismatch_raises(self):
        iface = self._make_iface()
        iface.baseline_checkpoint_sha256 = "abc123"
        with self.assertRaises(AssertionError):
            iface.verify(expected_b0_sha256="wrong_hash")


class TestSelectEventsFromAEC(unittest.TestCase):

    def test_empty_set_for_count0(self):
        from models.flash_vtg_gmr.event_cardinality import select_events_from_aec
        probs = torch.tensor([0.9, 0.05, 0.03, 0.01, 0.01])
        scores = torch.rand(10)
        mask = torch.ones(10, dtype=torch.bool)
        sel = select_events_from_aec(probs, scores, mask)
        self.assertEqual(len(sel), 0)

    def test_count1_returns_top1(self):
        from models.flash_vtg_gmr.event_cardinality import select_events_from_aec
        probs = torch.tensor([0.0, 0.9, 0.05, 0.03, 0.02])
        scores = torch.zeros(10)
        scores[3] = 0.99  # highest
        mask = torch.ones(10, dtype=torch.bool)
        sel = select_events_from_aec(probs, scores, mask)
        self.assertEqual(len(sel), 1)
        self.assertIn(3, sel)

    def test_count3_returns_top3(self):
        from models.flash_vtg_gmr.event_cardinality import select_events_from_aec
        probs = torch.tensor([0.0, 0.0, 0.0, 0.95, 0.05])
        scores = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0])
        mask = torch.ones(10, dtype=torch.bool)
        sel = select_events_from_aec(probs, scores, mask)
        self.assertEqual(len(sel), 3)
        self.assertIn(2, sel)   # highest
        self.assertIn(4, sel)   # second
        self.assertIn(1, sel)   # third

    def test_only_valid_modes_selected(self):
        from models.flash_vtg_gmr.event_cardinality import select_events_from_aec
        probs = torch.tensor([0.0, 0.9, 0.05, 0.03, 0.02])
        scores = torch.zeros(10)
        scores[7] = 0.99   # highest but invalid
        scores[2] = 0.80   # valid
        mask = torch.zeros(10, dtype=torch.bool)
        mask[2] = True
        sel = select_events_from_aec(probs, scores, mask)
        self.assertEqual(len(sel), 1)
        self.assertIn(2, sel)
        self.assertNotIn(7, sel)


if __name__ == "__main__":
    unittest.main()
