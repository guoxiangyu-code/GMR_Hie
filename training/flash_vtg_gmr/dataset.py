import torch
from torch.utils.data import Dataset
import numpy as np
import random
import hashlib
import logging
from os.path import join, exists
from models.flash_vtg_gmr.span_utils import span_xx_to_cxw
from models.flash_vtg_gmr.utils.basic_utils import load_jsonl, l2_normalize_np_array
from models.flash_vtg_gmr.utils.tensor_utils import pad_sequences_1d
from torchtext import vocab
import torch.nn as nn

logger = logging.getLogger(__name__)

TVSUM_SPLITS = {
    'BK': {
        'train': ['WxtbjNsCQ8A', 'EE-bNr36nyA', 'oDXZc0tZe04', 'uGu_10sucQo'],
        'val': ['Se3oxnaPsz0']
    },
    'BT': {
        'train': ['eQu1rNs0an0', 'qqR6AEXwxoQ', 'EYqVtI9YWJA', 'iVt07TCkFM0'],
        'val': ['JgHubY5Vw3Y']
    },
    'DS': {
        'train': ['kLxoNp-UchI', 'NyBmCxDoHJU', 'jcoYJXDG9sw', '-esJrBWj2d8'],
        'val': ['E11zDS9XGzg']
    },
    'FM': {
        'train': ['_xMr-HKMfVA', 'byxOvuiIJV0', 'VuWGsYPqAX8', 'xmEERLqJ2kU'],
        'val': ['JKpqYvAdIsw']
    },
    'GA': {
        'train': ['xxdtq8mxegs', 'i3wAGJaaktw', '0tmA_C6XwfM', '3eYKfiOEJNs'],
        'val': ['Bhxk-O1Y7Ho']
    },
    'MS': {
        'train': ['Hl-__g2gn_A', 'WG0MBPpPC6I', 'LRw_obCPUt0', '37rzWOQsNIw'],
        'val': ['Yi4Ij2NM7U4']
    },
    'PK': {
        'train': ['GsAD1KT1xo8', 'XkqCExn6_Us', 'b626MiF1ew4', 'PJrm840pAUI'],
        'val': ['cjibtmSLxQ4']
    },
    'PR': {
        'train': ['RBCABdttQmI', 'z_6gVvQb2d0', '4wU_LUjG5Ic', '91IHQYk1IQM'],
        'val': ['fWutDQy1nnY']
    },
    'VT': {
        'train': ['gzDbaEs1Rlg', 'XzYM3PfTM4w', '98MoyGZKHXc', 'AwmHb44_ouw'],
        'val': ['J0nA4VgnoCo']
    },
    'VU': {
        'train': ['akI8YFjEmUw', 'HT5vyqe0Xaw', 'vdmoEJ5YbrQ', 'xwqBXPGE9pQ'],
        'val': ['sTEELN-vY30']
    }
}
class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state", "features"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None, mr_only=False, keep_empty_gt=False,
                 strict_data_contract=False, require_text_mask=False,
                 text_store_length=77, seed=0, split=None,
                 legacy_text_mask=False, legacy_gt_sampling=False):
        self.dset_name = dset_name
        self.data_path = data_path
        # MR-only mode does not require highlight-detection annotations.
        self.mr_only = mr_only
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        if max_v_l == -1:
            max_v_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        # If True, keep samples with empty GT windows (negative samples).
        self.keep_empty_gt = bool(keep_empty_gt)
        self.strict_data_contract = bool(strict_data_contract)
        self.require_text_mask = bool(require_text_mask)
        self.legacy_text_mask = bool(legacy_text_mask)
        self.legacy_gt_sampling = bool(legacy_gt_sampling)
        self.text_store_length = int(text_store_length)
        self.seed = int(seed)
        self.split = split or self._infer_split(data_path)
        self._epoch = 0
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        # load specific domain data for tvsum dataset
        if self.dset_name in ['tvsum', 'tvsum_sfc']:
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data

        # load specific domain data for youtube-hl dataset
        if self.dset_name == 'youtube_uni':
            target_domain = dset_domain
            assert target_domain in ["dog", "gymnastics", "parkour", "skating", "skiing", "surfing"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data

        self.use_glove = False
        self.use_glove = 'vgg' in self.v_feat_dirs[0]

        if self.dset_name == 'charadesSTA' and self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

        # Load all data into memory
        self._preload_data()

    def _vid_to_stem(self, vid):
        """Strip video extension (.mp4/.mkv/.webm/.avi/.mov/.m4v) to get feature file stem."""
        if not vid:
            return vid
        for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"):
            if vid.endswith(ext):
                return vid[: -len(ext)]
        return vid

    @staticmethod
    def _infer_split(data_path):
        stem = str(data_path).rsplit("/", 1)[-1].split(".", 1)[0].lower()
        return stem if stem in {"train", "val", "test"} else "unknown"

    def _has_video_and_query_features(self, d):
        vid = d.get("vid")
        qid = d.get("qid")
        stem = self._vid_to_stem(vid) if vid else None
        if self.use_video and self.v_feat_dirs:
            for feat_dir in self.v_feat_dirs:
                if exists(join(feat_dir, f"{stem}.npz")):
                    continue
                if exists(join(feat_dir, f"{stem}.npy")):
                    continue
                if exists(join(feat_dir, f"{stem}.pt")):
                    continue
                return False
        if self.q_feat_dir and qid is not None:
            if not exists(join(self.q_feat_dir, f"qid{qid}.npz")):
                return False
        return True

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        # Optional: filter empty GT windows when training MR-only without existence head.
        if self.load_labels and (not self.keep_empty_gt):
            filtered = []
            for d in datalist:
                windows = d.get("relevant_windows", None)
                if windows is None:
                    filtered.append(d)
                    continue
                if isinstance(windows, list) and len(windows) > 0:
                    filtered.append(d)
            datalist = filtered
        # Skip records whose video or query features are unavailable.
        kept = [d for d in datalist if self._has_video_and_query_features(d)]
        if len(kept) < len(datalist):
            if self.strict_data_contract:
                missing = [
                    (d.get("qid"), d.get("vid"))
                    for d in datalist
                    if not self._has_video_and_query_features(d)
                ]
                raise FileNotFoundError(
                    f"Strict data contract forbids missing features; first missing rows: {missing[:5]}"
                )
            logger.warning(
                "[FlashVTG] Skip missing features: {} examples removed, {} kept.".format(
                    len(datalist) - len(kept), len(kept)
                )
            )
            datalist = kept
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def _preload_data(self):
        self.preloaded_data = []
        for index in range(len(self.data)):
            meta = self.data[index]
            model_inputs = self._load_model_inputs(meta)
            self.preloaded_data.append((meta, model_inputs))

    def _load_model_inputs(self, meta):
        model_inputs = dict()

        if self.use_glove:
            model_inputs["query_feat"] = self.get_query(meta["query"])
            model_inputs["query_mask"] = torch.ones(
                len(model_inputs["query_feat"]), dtype=torch.bool
            )
        else:
            query_feat, query_mask = self._get_query_feat_by_qid(meta["qid"])
            model_inputs["query_feat"] = query_feat
            model_inputs["query_mask"] = query_mask

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.dset_name in ['tvsum']:
            model_inputs["span_labels"] = torch.tensor([[0., 0.]])
            meta_label = meta['label']
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            if len(model_inputs["saliency_all_labels"]) != len(model_inputs["video_feat"]):
                model_inputs["video_feat"] = model_inputs["video_feat"][:len(model_inputs["saliency_all_labels"])]

        elif self.dset_name == 'youtube_uni':
            model_inputs["span_labels"] = torch.tensor([[0., 0.]])
            meta_label = meta['label']
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all_youtube(meta_label, ctx_l)
        else:
            if "relevant_windows" in meta: ## For Qvhighlights test set
                windows = meta.get("relevant_windows", [])
                exist = 1.0 if (isinstance(windows, list) and len(windows) > 0) else 0.0
                model_inputs["exist_label"] = float(exist)

                model_inputs["span_labels"] = self.get_span_labels(
                    windows, ctx_l, meta=meta
                )  # (#windows, 2) or (0,2)

                # Build a minimal saliency supervision that is safe for empty windows.
                # For negative samples, we create all-zero saliency labels.
                if not (isinstance(windows, list) and len(windows) > 0):
                    model_inputs["saliency_all_labels"] = np.zeros(ctx_l, dtype=np.float32)
                    # Use identical indices so pairwise margin loss has zero gradient.
                    model_inputs["saliency_pos_labels"] = [0, 0]
                    model_inputs["saliency_neg_labels"] = [0, 0]
                else:
                    if self.dset_name in ['charadesSTA', 'tacos', 'activitynet']: ## charades, tacos, nlq
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_sub_as_query(
                                windows[0], meta["duration"], ctx_l, rng=self._local_rng(meta)
                            )  # only one gt
                    elif self.dset_name in ['nlq']:
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_sub_as_query(windows[0], meta["duration"], ctx_l, 2)  # only one gt
                    elif "subs_train" not in self.data_path:
                        # MR-only mode does not consume QVHighlights saliency fields.
                        if getattr(self, "mr_only", False) or ("relevant_clip_ids" not in meta) or ("saliency_scores" not in meta):
                            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_sub_as_query(
                                    windows[0], meta["duration"], ctx_l, rng=self._local_rng(meta)
                                )
                        else:
                            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                                self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                    else:
                        model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_sub_as_query(
                                windows[0], meta["duration"], ctx_l, rng=self._local_rng(meta)
                            )  # only one gt

        if 'qvhighlight' in self.data_path:
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]
        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return model_inputs

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, index):
        meta, cached_inputs = self.preloaded_data[index]
        if not (self.strict_data_contract and self.mr_only):
            return meta, cached_inputs
        model_inputs = dict(cached_inputs)
        windows = meta.get("relevant_windows", [])
        ctx_l = len(model_inputs["video_feat"])
        if windows:
            pos, neg, scores = self.get_saliency_labels_sub_as_query(
                windows[0], meta["duration"], ctx_l, rng=self._local_rng(meta)
            )
            model_inputs["saliency_pos_labels"] = pos
            model_inputs["saliency_neg_labels"] = neg
            model_inputs["saliency_all_labels"] = scores
        return meta, model_inputs

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def _local_rng(self, meta):
        source = meta.get("source", meta.get("dataset_source", "unknown"))
        identity = (
            f"{self.seed}|{self.split}|{self._epoch}|{source}|"
            f"{meta.get('qid')}|{self._vid_to_stem(meta.get('vid'))}"
        )
        local_seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
        return random.Random(local_seed)

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2, rng=None):
        rng = rng or random
        clip_len = self.clip_len if self.strict_data_contract else duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            population = list(range(gt_st, gt_ed + 1))
            if len(population) >= max_n:
                pos_clip_indices = rng.sample(population, k=max_n)
            else:
                pos_clip_indices = [population[0]] * max_n
        else:
            if self.dset_name == 'nlq':
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        try:
            neg_clip_indices = rng.sample(neg_pool, k=max_n)
        except ValueError:
            neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array


    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):

        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_youtube(self, labels, ctx_l, max_n=1, add_easy_negative=False):

        # Youtube-hl only have binary score
        agg_scores = np.array(labels)[:, 0] # (L, 1) --> (L, )
        score_array = agg_scores * 1

        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array


    def get_span_labels(self, windows, ctx_l, meta=None):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            `self.max_windows=-1` keeps every window. Positive values keep the
            first windows in canonical manifest order.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        # Support empty windows (negative samples). Return an empty (0,2) tensor with correct dtype.
        if windows is None or (isinstance(windows, (list, tuple)) and len(windows) == 0):
            if self.span_loss_type == "l1":
                return torch.zeros((0, 2), dtype=torch.float32)
            elif self.span_loss_type == "ce":
                return torch.zeros((0, 2), dtype=torch.long)
            else:
                raise NotImplementedError
        windows = [list(window) for window in windows]
        if self.max_windows != -1 and len(windows) > self.max_windows:
            if self.legacy_gt_sampling:
                if meta is None:
                    raise ValueError("legacy GT sampling requires sample metadata")
                self._local_rng(meta).shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            feature = torch.from_numpy(q_feat['last_hidden_state'])
            return feature, torch.ones(len(feature), dtype=torch.bool)
            # return torch.from_numpy(q_feat['token'])
        # youtube-hl
        elif self.dset_name == 'youtube_uni':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid)))
            feature = torch.from_numpy(q_feat['last_hidden_state'])
            return feature, torch.ones(len(feature), dtype=torch.bool)

        elif self.dset_name in ['tacos', 'nlq']:
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            return torch.from_numpy(q_feat), torch.ones(len(q_feat), dtype=torch.bool)
        else:
            q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            if exists(q_feat_path):
                with np.load(q_feat_path, allow_pickle=False) as archive:
                    if self.q_feat_type not in archive.files:
                        raise KeyError(f"{q_feat_path} lacks key {self.q_feat_type!r}")
                    q_feat = archive[self.q_feat_type].astype(np.float32)
                    stored_mask = (
                        archive["attention_mask"].copy()
                        if "attention_mask" in archive.files and not self.legacy_text_mask
                        else None
                    )
            elif not self.strict_data_contract:
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.pt")
                q_feat = torch.load(q_feat_path).float().numpy()
                stored_mask = None
            else:
                raise FileNotFoundError(q_feat_path)

            if self.strict_data_contract and self.q_feat_type == "last_hidden_state":
                if q_feat.shape != (self.text_store_length, q_feat.shape[-1]):
                    raise ValueError(
                        f"{q_feat_path}: expected {self.text_store_length} stored rows, got {q_feat.shape}"
                    )
            if self.require_text_mask and stored_mask is None:
                raise KeyError(f"{q_feat_path}: strict text contract requires attention_mask")
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
                if stored_mask is None:
                    query_mask = np.ones(len(q_feat), dtype=bool)
                else:
                    stored_mask = np.asarray(stored_mask)
                    if stored_mask.shape != (self.text_store_length,):
                        raise ValueError(
                            f"{q_feat_path}: invalid attention_mask shape {stored_mask.shape}"
                        )
                    query_mask = stored_mask[:self.max_q_l].astype(bool)
                    valid_length = int(stored_mask.astype(bool).sum())
                    expected = np.arange(self.text_store_length) < valid_length
                    if not np.array_equal(stored_mask.astype(bool), expected):
                        raise ValueError(f"{q_feat_path}: non-contiguous attention_mask")
                    if valid_length > self.max_q_l:
                        raise ValueError(
                            f"{q_feat_path}: valid length {valid_length} exceeds max_q_l={self.max_q_l}"
                        )
            else:
                query_mask = np.ones(len(q_feat), dtype=bool)
            if self.strict_data_contract and not np.isfinite(q_feat).all():
                raise ValueError(f"{q_feat_path}: text feature contains NaN or Inf")
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            q_feat[~query_mask] = 0
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat), torch.from_numpy(query_mask)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                try:
                    _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                    _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                    _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                    _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                    _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                except:
                    try:
                        _feat_path = join(_feat_dir, f"{vid}.npy")
                        _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                    except:
                        _feat_path = join(_feat_dir, f"{vid}.npz")
                        _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)

                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            lengths = [len(e) for e in v_feat_list]
            if self.strict_data_contract and len(set(lengths)) != 1:
                raise ValueError(f"Video stream length mismatch for {vid}: {lengths}")
            min_len = min(lengths)
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        elif self.dset_name == 'youtube_uni':
            vid_for_path = self._vid_to_stem(vid) if vid else vid
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                # Only single npz files per directory
                try:
                    _feat_path = join(_feat_dir, f"{vid_for_path}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                except:
                    _feat_path = join(_feat_dir, f"{vid_for_path}.npy")
                    _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list] # TODO do we need to cut the length over the min_len?
            v_feat = np.concatenate(v_feat_list, axis=1)

        else:
            v_feat_list = []
            # Feature files are keyed by the video stem, without a media extension.
            vid_for_path = self._vid_to_stem(vid) if vid else vid
            for _feat_dir in self.v_feat_dirs:
                if self.strict_data_contract:
                    _feat_path = join(_feat_dir, f"{vid_for_path}.npz")
                    if not exists(_feat_path):
                        raise FileNotFoundError(_feat_path)
                    with np.load(_feat_path, allow_pickle=False) as archive:
                        if set(archive.files) != {"features"}:
                            raise ValueError(
                                f"{_feat_path}: expected only key 'features', got {archive.files}"
                            )
                        _feat = archive["features"][:self.max_v_l].astype(np.float32)
                else:
                    try:
                        _feat_path = join(_feat_dir, f"{vid_for_path}.npz")
                        _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                    except:
                        try:
                            _feat_path = join(_feat_dir, f"{vid_for_path}.pt")
                            _feat = torch.load(_feat_path)[:self.max_v_l].float().numpy()
                        except:
                            _feat_path = join(_feat_dir, f"{vid_for_path}.npy")
                            _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                if self.strict_data_contract:
                    if _feat.ndim != 2 or len(_feat) == 0:
                        raise ValueError(f"{_feat_path}: expected a non-empty 2-D feature")
                    if not np.isfinite(_feat).all():
                        raise ValueError(f"{_feat_path}: contains NaN or Inf")
                    if np.any(np.linalg.norm(_feat, axis=1) <= 1e-12):
                        raise ValueError(f"{_feat_path}: contains a zero-norm real row")
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            lengths = [len(e) for e in v_feat_list]
            if self.strict_data_contract and len(set(lengths)) != 1:
                raise ValueError(f"Video stream length mismatch for {vid}: {lengths}")
            min_len = min(lengths)
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
            if self.strict_data_contract and not np.isfinite(v_feat).all():
                raise ValueError(f"Concatenated video feature contains NaN or Inf for {vid}")
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    # batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?
    batch_meta = [e[0] for e in batch]  # seems no need to collate ?

    model_inputs_keys = set(batch[0][1].keys())
    for e in batch[1:]:
        model_inputs_keys &= set(e[1].keys())
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e[1]["span_labels"]) for e in batch]
            continue
        if k == "exist_label":
            batched_data[k] = torch.tensor([e[1][k] for e in batch], dtype=torch.float32)
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e[1][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e[1][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == "query_mask":
            masks = [e[1][k].bool() for e in batch]
            if len({tuple(mask.shape) for mask in masks}) != 1:
                raise ValueError("query_mask must have a fixed shape within a batch")
            batched_data[k] = torch.stack(masks, dim=0)
            continue
        if k == 'qid':
            batched_data[k] = [e[1][k] for e in batch]
            continue
        if k == 'vid':
            batched_data[k] = [e[1][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e[1][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs.get(
            "query_mask", batched_model_inputs["query_feat"][1]
        ).to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        vid=batched_model_inputs["vid"],
        qid=batched_model_inputs["qid"],
    )
    targets = {}

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "exist_label" in batched_model_inputs:
        targets["exist_label"] = batched_model_inputs["exist_label"].to(device, non_blocking=non_blocking)

    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
