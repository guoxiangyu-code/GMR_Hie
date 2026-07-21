import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from training.flash_vtg_gmr.contracts import (
    canonical_json_sha256,
    query_key,
    stable_exact_deduplicate_windows,
    validate_encoder_mask,
)
from training.flash_vtg_gmr.dataset import (
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)
from training.flash_vtg_gmr.reproducibility import (
    capture_rng_state,
    make_data_generator,
    restore_rng_state,
)


class Part1ContractTest(unittest.TestCase):
    def test_maintained_gmr_cls_is_used_by_training(self):
        from training.flash_vtg_gmr.inference import _compute_legacy_cls_summary

        submission = [
            {"qid": 1, "pred_exist_score": 0.9},
            {"qid": 2, "pred_exist_score": 0.1},
        ]
        ground_truth = [
            {"qid": 1, "relevant_windows": [[0.0, 1.0]]},
            {"qid": 2, "relevant_windows": []},
        ]
        self.assertEqual(
            _compute_legacy_cls_summary(submission, ground_truth, 0.5),
            {"TPR": 100.0, "TNR": 100.0, "BalancedAcc": 100.0},
        )

    def test_full_evaluator_rejects_partial_qid_coverage(self):
        from eval.eval_main import evaluate_gmr

        submission = [{"qid": 1, "pred_relevant_windows": [], "pred_exist_score": 0.0}]
        ground_truth = [
            {"qid": 1, "relevant_windows": []},
            {"qid": 2, "relevant_windows": [[0.0, 1.0]]},
        ]
        with self.assertRaisesRegex(ValueError, "qid coverage mismatch"):
            evaluate_gmr(submission, ground_truth, verbose=False, map_num_workers=1)

    def test_full_evaluator_reports_cardinality_groups(self):
        from eval.eval_main import evaluate_gmr

        submission = [
            {"qid": 1, "pred_relevant_windows": [], "pred_exist_score": 0.1},
            {
                "qid": 2,
                "pred_relevant_windows": [[0.0, 1.0, 0.9]],
                "pred_exist_score": 0.9,
            },
            {
                "qid": 3,
                "pred_relevant_windows": [[0.0, 1.0, 0.9], [2.0, 3.0, 0.8]],
                "pred_exist_score": 0.9,
            },
        ]
        ground_truth = [
            {"qid": 1, "relevant_windows": []},
            {"qid": 2, "relevant_windows": [[0.0, 1.0]]},
            {"qid": 3, "relevant_windows": [[0.0, 1.0], [2.0, 3.0]]},
        ]
        result = evaluate_gmr(
            submission, ground_truth, verbose=False, map_num_workers=1
        )
        self.assertEqual(result["grouped"]["null"]["FPR@0.40"], 0.0)
        self.assertEqual(result["grouped"]["single"]["G-mIoU@1"], 100.0)
        self.assertEqual(result["grouped"]["multi"]["G-mIoU@3"], 100.0)

    def test_canonical_tensor_hash_handles_scalar_state(self):
        from training.flash_vtg_gmr.finalize_baselines import tensor_state_sha256

        state = {"scalar": torch.tensor(1.0), "vector": torch.arange(3)}
        self.assertEqual(tensor_state_sha256(state), tensor_state_sha256(state))
        changed = {**state, "scalar": torch.tensor(2.0)}
        self.assertNotEqual(tensor_state_sha256(state), tensor_state_sha256(changed))

    def test_baseline_finalizer_validates_manifest_content_hash(self):
        from training.flash_vtg_gmr.finalize_baselines import (
            validate_content_hashed_json,
        )

        payload = {"setting": "f-lighthouse"}
        path = self.root / "hashed_manifest.json"
        path.write_text(
            json.dumps({**payload, "content_sha256": canonical_json_sha256(payload)}),
            encoding="utf-8",
        )
        self.assertEqual(validate_content_hashed_json(path)["setting"], "f-lighthouse")
        path.write_text(
            json.dumps({**payload, "content_sha256": "wrong"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Content SHA256 mismatch"):
            validate_content_hashed_json(path)

    def test_lighthouse_text_resume_requires_saved_input_ids(self):
        from training.flash_vtg_gmr.extract_lighthouse_features import (
            existing_text_is_valid,
        )

        path = self.root / "lighthouse_text.npz"
        hidden = np.ones((77, 512), dtype=np.float32)
        mask = np.zeros(77, dtype=np.float32)
        mask[:4] = 1
        np.savez(path, last_hidden_state=hidden, attention_mask=mask)
        self.assertFalse(existing_text_is_valid(path))
        input_ids = np.zeros(77, dtype=np.int64)
        input_ids[:4] = [49406, 1, 2, 49407]
        np.savez(
            path,
            last_hidden_state=hidden,
            attention_mask=mask,
            input_ids=input_ids,
        )
        self.assertTrue(existing_text_is_valid(path))
        input_ids[3] = 0
        np.savez(
            path,
            last_hidden_state=hidden,
            attention_mask=mask,
            input_ids=input_ids,
        )
        self.assertFalse(existing_text_is_valid(path))

    def test_lighthouse_text_flat_qid_collision_includes_source(self):
        from training.flash_vtg_gmr.extract_lighthouse_features import query_inventory

        rows = [
            {
                "qid": 1,
                "query": "same query",
                "dataset_source": "sportsmoments",
            },
            {
                "qid": 1,
                "query": "same query",
                "dataset_source": "worldcup2022",
            },
        ]
        with self.assertRaisesRegex(ValueError, "Flat text filename collision"):
            query_inventory(rows)

    def test_formal_feature_gate_rejects_f_old(self):
        from training.flash_vtg_gmr.config import BaseOptions
        from training.flash_vtg_gmr.contracts import canonical_json_sha256

        payload = {"concat_order": ["slowfast", "clip"], "setting": "f-old"}
        manifest = {**payload, "content_sha256": canonical_json_sha256(payload)}
        path = self.root / "feature_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "setting=f-lighthouse"):
            BaseOptions._validate_feature_manifest(
                SimpleNamespace(feature_manifest=str(path))
            )

    def test_formal_feature_gate_rejects_training_mode_encoder(self):
        from training.flash_vtg_gmr.config import BaseOptions

        payload = {
            "concat_order": ["slowfast", "clip"],
            "setting": "f-lighthouse",
            "stream_dimensions": {"slowfast": 2304, "clip": 512},
            "video_feature_dim": 2816,
            "text_feature_dim": 512,
            "per_stream_normalization": True,
            "normalization_eps": 1e-5,
            "known_provenance": True,
            "encoder_mode": "train",
            "text_token_alignment_status": "verified",
            "cross_stream_length_policy": (
                "lighthouse-trim-shorter-at-extraction; exact-or-fail-loader"
            ),
        }
        manifest = {**payload, "content_sha256": canonical_json_sha256(payload)}
        path = self.root / "train_mode_feature_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "encoders in eval mode"):
            BaseOptions._validate_feature_manifest(
                SimpleNamespace(feature_manifest=str(path))
            )

    def test_lighthouse_batch_invariance_audit(self):
        from training.flash_vtg_gmr.audit_features import _audit_batch_invariance

        reference = self.root / "batch_reference"
        canary = self.root / "batch_canary"
        for root in (reference, canary):
            (root / "clip").mkdir(parents=True)
            (root / "slowfast").mkdir(parents=True)
        for index in range(2):
            for stream, dimension in (("clip", 512), ("slowfast", 2304)):
                features = self.rng.normal(size=(3, dimension)).astype(np.float32)
                for root in (reference, canary):
                    np.savez(root / stream / f"video_{index}.npz", features=features)
        result = _audit_batch_invariance(reference, canary, minimum_videos=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["shared_video_count"], 2)
        self.assertGreaterEqual(result["streams"]["slowfast"]["row_cosine_min"], 0.99999)

    def test_balanced_canary_selection_covers_both_sources(self):
        from training.flash_vtg_gmr.extract_lighthouse_features import select_shard

        items = [
            {"source": "sportsmoments", "stem": f"s{index}"}
            for index in range(60)
        ] + [
            {"source": "worldcup2022", "stem": f"w{index}"}
            for index in range(60)
        ]
        selected = select_shard(
            items, 0, 1, 50, balanced_sources=True
        )
        counts = {
            source: sum(item["source"] == source for item in selected)
            for source in ("sportsmoments", "worldcup2022")
        }
        self.assertEqual(counts, {"sportsmoments": 25, "worldcup2022": 25})

    def test_manifest_builder_requires_explicit_safety_flags(self):
        from training.flash_vtg_gmr.build_manifests import build

        with self.assertRaisesRegex(ValueError, "--no_truncate"):
            build(SimpleNamespace(no_truncate=False, audit_split_identity=True))
        with self.assertRaisesRegex(ValueError, "--audit_split_identity"):
            build(SimpleNamespace(no_truncate=True, audit_split_identity=False))

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.slowfast = self.root / "slowfast"
        self.clip = self.root / "clip"
        self.text = self.root / "clip_text"
        for directory in (self.slowfast, self.clip, self.text):
            directory.mkdir()
        self.rng = np.random.default_rng(7)
        self.rows = []
        counts = [0, 1, 6, 7]
        for qid, count in enumerate(counts, start=1):
            vid = f"video_{qid}"
            windows = [[float(index), float(index + 1)] for index in range(count)]
            self.rows.append(
                {
                    "qid": qid,
                    "vid": vid,
                    "query": f"query {qid}",
                    "duration": 20.0,
                    "dataset_source": "worldcup2022",
                    "relevant_windows": windows,
                }
            )
            self._write_video(vid, length=10)
            self._write_text(qid, padding_value=3.0)
        self.data_path = self.root / "train.jsonl"
        self.data_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_video(self, vid, length):
        slowfast = self.rng.normal(size=(length, 2304)).astype(np.float32)
        clip = self.rng.normal(size=(length, 512)).astype(np.float32)
        np.savez(self.slowfast / f"{vid}.npz", features=slowfast)
        np.savez(self.clip / f"{vid}.npz", features=clip)

    def _write_text(self, qid, padding_value):
        hidden = self.rng.normal(size=(77, 512)).astype(np.float32)
        hidden[6:] = padding_value
        mask = np.zeros(77, dtype=np.float32)
        mask[:6] = 1
        np.savez(
            self.text / f"qid{qid}.npz",
            last_hidden_state=hidden,
            attention_mask=mask,
        )

    def _dataset(self):
        return StartEndDataset(
            dset_name="hl",
            data_path=str(self.data_path),
            v_feat_dirs=[str(self.slowfast), str(self.clip)],
            q_feat_dir=str(self.text),
            max_q_l=40,
            max_v_l=75,
            ctx_mode="video_tef",
            clip_len=2,
            max_windows=-1,
            mr_only=True,
            keep_empty_gt=True,
            strict_data_contract=True,
            require_text_mask=True,
            text_store_length=77,
            seed=2024,
            split="train",
        )

    def test_exact_deduplication_preserves_distinct_overlaps(self):
        windows = [[1, 3], [1, 3], [1.0, 3.001], [2, 4]]
        self.assertEqual(
            stable_exact_deduplicate_windows(windows),
            [[1.0, 3.0], [1.0, 3.001], [2.0, 4.0]],
        )

    def test_legacy_gt_cap_uses_copy_and_local_rng(self):
        dataset = self._dataset()
        dataset.max_windows = 5
        dataset.legacy_gt_sampling = True
        meta = self.rows[-1]
        original = [window[:] for window in meta["relevant_windows"]]
        first = dataset.get_span_labels(meta["relevant_windows"], 10, meta=meta)
        second = dataset.get_span_labels(meta["relevant_windows"], 10, meta=meta)
        self.assertEqual(len(first), 5)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(meta["relevant_windows"], original)

    def test_mask_contract(self):
        mask = np.zeros(77)
        mask[:9] = 1
        validated, length = validate_encoder_mask(mask, 77, 40)
        self.assertEqual(length, 9)
        self.assertTrue(validated[:9].all())
        mask[10] = 1
        with self.assertRaises(ValueError):
            validate_encoder_mask(mask, 77, 40)

    def test_dataset_setting_is_part_of_query_identity(self):
        standard = query_key("standard", "train", "worldcup2022", 853)
        full = query_key("full", "train", "worldcup2022", 853)
        self.assertNotEqual(standard, full)

    def test_extreme_counts_real_mask_and_padding_isolation(self):
        dataset = self._dataset()
        self.assertEqual(
            [tuple(inputs["span_labels"].shape) for _, inputs in dataset],
            [(0, 2), (1, 2), (6, 2), (7, 2)],
        )
        _, first = dataset[0]
        self.assertEqual(tuple(first["query_feat"].shape), (40, 512))
        self.assertEqual(first["query_mask"].tolist(), [True] * 6 + [False] * 34)
        self.assertTrue(torch.equal(first["query_feat"][6:], torch.zeros(34, 512)))

        batch = start_end_collate([dataset[index] for index in range(4)])
        model_inputs, targets = prepare_batch_inputs(batch[1], torch.device("cpu"))
        self.assertEqual(tuple(model_inputs["src_txt"].shape), (4, 40, 512))
        self.assertEqual(tuple(model_inputs["src_txt_mask"].shape), (4, 40))
        self.assertEqual([len(item["spans"]) for item in targets["span_labels"]], [0, 1, 6, 7])
        self.assertTrue(torch.isfinite(model_inputs["src_vid"]).all())

    def test_video_stream_normalization_and_concat_order(self):
        _, inputs = self._dataset()[0]
        with np.load(self.slowfast / "video_1.npz") as archive:
            slowfast = archive["features"]
        with np.load(self.clip / "video_1.npz") as archive:
            clip = archive["features"]
        slowfast = slowfast / (
            np.linalg.norm(slowfast, axis=-1, keepdims=True) + 1e-5
        )
        clip = clip / (np.linalg.norm(clip, axis=-1, keepdims=True) + 1e-5)
        expected = torch.from_numpy(np.concatenate([slowfast, clip], axis=1))
        self.assertTrue(torch.equal(inputs["video_feat"][:, :2816], expected))

    def test_padding_perturbation_does_not_change_loaded_text(self):
        before = self._dataset()[0][1]
        path = self.text / "qid1.npz"
        with np.load(path) as archive:
            hidden = archive["last_hidden_state"].copy()
            mask = archive["attention_mask"].copy()
        hidden[6:] = -1000.0
        np.savez(path, last_hidden_state=hidden, attention_mask=mask)
        after = self._dataset()[0][1]
        self.assertTrue(torch.equal(before["query_mask"], after["query_mask"]))
        self.assertTrue(torch.allclose(before["query_feat"], after["query_feat"]))

    def test_cross_stream_length_mismatch_fails(self):
        clip = self.rng.normal(size=(9, 512)).astype(np.float32)
        np.savez(self.clip / "video_1.npz", features=clip)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self._dataset()

    def test_epoch_local_rng_is_replayable(self):
        dataset = self._dataset()
        dataset.set_epoch(3)
        labels_a = dataset[1][1]["saliency_pos_labels"]
        labels_b = dataset[1][1]["saliency_pos_labels"]
        self.assertEqual(labels_a, labels_b)

    def test_extreme_counts_model_criterion_backward(self):
        import nncore
        from models.flash_vtg_gmr.model import build_model1

        dataset = self._dataset()
        batch = start_end_collate([dataset[index] for index in range(4)])
        model_inputs, targets = prepare_batch_inputs(batch[1], torch.device("cpu"))
        targets["label"] = batch[0]
        targets["fps"] = torch.full((4,), 0.5)

        options = json.loads(Path("configs/flash_vtg_gmr/soccer_gmr.json").read_text())
        options.update(
            {
                "device": torch.device("cpu"),
                "v_feat_dim": 2818,
                "train_path": str(self.data_path),
                "eval_path": None,
                "max_windows": -1,
                "use_neg": False,
                "mr_only": True,
                "use_exist_head": True,
            }
        )
        args = SimpleNamespace(**options)
        args.cfg = nncore.Config.from_file("configs/flash_vtg_gmr/model.py")
        model, criterion = build_model1(args)
        model.train()
        outputs = model(**model_inputs, targets=targets)
        loss_dict = criterion(batch, outputs, targets)
        weighted = sum(
            value * criterion.weight_dict[name]
            for name, value in loss_dict.items()
            if name in criterion.weight_dict
        )
        self.assertTrue(torch.isfinite(weighted).all())
        weighted.backward()
        self.assertIsNotNone(model.exist_head[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.exist_head[-1].weight.grad).all())

    def test_generator_state_resume(self):
        generator = make_data_generator(2024)
        _ = torch.randperm(20, generator=generator)
        state = capture_rng_state(generator)
        expected = torch.randperm(20, generator=generator)
        restored = make_data_generator(0)
        restore_rng_state(state, restored)
        actual = torch.randperm(20, generator=restored)
        self.assertTrue(torch.equal(expected, actual))

    def test_prediction_clamp_uses_decode_duration(self):
        from training.flash_vtg_gmr.inference import clamp_prediction_to_decode_duration

        prediction = {
            "_decode_duration": 5.5,
            "pred_relevant_windows": [
                [4.0, 6.0, 0.9],
                [5.5, 8.0, 0.8],
                [-1.0, 1.0, 0.7],
            ],
        }
        result = clamp_prediction_to_decode_duration(prediction)
        self.assertEqual(
            result["pred_relevant_windows"],
            [[4.0, 5.5, 0.9], [0.0, 1.0, 0.7]],
        )
        self.assertNotIn("_decode_duration", result)

    def test_generated_manifest_hashes_when_present(self):
        path = Path("artifacts/manifests/standard/manifest_index.json")
        if not path.is_file():
            self.skipTest("full corpus artifacts were not generated")
        document = json.loads(path.read_text(encoding="utf-8"))
        content_hash = document.pop("content_sha256")
        self.assertEqual(content_hash, canonical_json_sha256(document))
        self.assertEqual(document["data_manifests"]["train"]["row_count"], 4138)
        self.assertEqual(document["data_manifests"]["val"]["row_count"], 465)
        self.assertEqual(document["data_manifests"]["test"]["row_count"], 1036)


if __name__ == "__main__":
    unittest.main()
