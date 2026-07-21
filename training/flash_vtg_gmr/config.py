import os
import time
import torch
import argparse

from models.flash_vtg_gmr.utils.basic_utils import (
    dict_to_markdown,
    load_json,
    make_zipfile,
    mkdirp,
    save_json,
)
import shutil
import json
import platform
import sys

from training.flash_vtg_gmr.contracts import canonical_json_sha256, sha256_file

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("config", help="config file")
        parser.add_argument("--use_neg", action="store_true", help="use negative samples in training")
        parser.add_argument("--dset_name", type=str, choices=["hl", 'tvsum', 'charadesSTA', 'tacos','youtube_uni', 'qv_internvideo2', 'charadesSTA_internvideo2'])
        parser.add_argument("--dset_domain", type=str,
                            help="Domain to train for tvsum dataset. (Only used for tvsum and youtube-hl)")

        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--data_ratio", type=float, default=1.0,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2024, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=400, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--n_epoch", type=int, default=700, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.set_defaults(drop_last=False)
        parser.add_argument("--drop_last", dest="drop_last", action="store_true",
                            help="Drop an incomplete final training batch")
        parser.add_argument("--no_drop_last", dest="drop_last", action="store_false",
                            help="Keep an incomplete final training batch")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--eval_epoch", type=int, default=2,
                            help="inference epoch")
        parser.add_argument("--eval_full_only", action="store_true",
                            help="Only evaluate and log full-range MR metrics during validation.")
        parser.add_argument("--mr_only", action="store_true",
                            help="Only run Moment Retrieval (MR). Skip Highlight Detection (HL) related labels/metrics.")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # Data config
        parser.add_argument("--max_q_l", type=int, default=-1)
        parser.add_argument("--max_v_l", type=int, default=-1)
        parser.add_argument("--clip_length", type=float, default=2)
        parser.add_argument("--max_windows", type=int, default=5)
        parser.add_argument("--strict_data_contract", action="store_true",
                            help="Require the frozen F-Lighthouse mask/shape/length contract")
        parser.add_argument("--require_text_mask", action="store_true",
                            help="Require NPZ attention_mask instead of synthesizing one")
        parser.add_argument("--legacy_text_mask", action="store_true",
                            help="Diagnostic only: ignore the stored text mask")
        parser.add_argument("--legacy_gt_sampling", action="store_true",
                            help="Diagnostic only: apply the legacy randomized GT cap to a copy")
        parser.add_argument("--text_store_length", type=int, default=77)
        parser.add_argument("--feature_manifest", type=str, default=None)
        parser.add_argument("--data_manifest_index", type=str, default=None)
        parser.add_argument(
            "--baseline_variant",
            choices=("B0-legacy", "B0-mask-only", "B0-gt-only", "B0"),
            default="B0-legacy",
        )
        parser.add_argument("--repro_check", action="store_true")
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=-1,
            help="Diagnostic cap on optimizer updates per epoch; -1 disables it",
        )

        parser.add_argument("--train_path", type=str, default=None)
        parser.add_argument("--eval_path", type=str, default=None,
                            help="Evaluating during training, for Dev set. If None, will only do training, ")
        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
        parser.add_argument("--a_feat_dir", type=str, help="audio feature dir")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument("--a_feat_dim", type=int, help="audio feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")
        parser.add_argument("--q_feat_type", type=str, default="last_hidden_state", help="use video features")

        # Model config
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        parser.add_argument('--kernel_size', default=3, type=int,
                            help="Number of ")
        parser.add_argument('--num_conv_layers', default=3, type=int,
                            help="Number of ")
        parser.add_argument('--num_mlp_layers', default=3, type=int,
                            help="Number of ")
        parser.add_argument('--use_SRM', action="store_true")

        # * Transformer
        parser.add_argument('--enc_layers', default=3, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--t2v_layers', default=2, type=int,
                            help="Number of ACA layers in the transformer")
        parser.add_argument('--dummy_layers', default=2, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_dummies', default=45, type=int,
                            help="Number of dummy tokens")
        parser.add_argument('--total_prompts', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--num_prompts', default=1, type=int,
                            help="Number of dummy tokens")
        parser.add_argument('--pre_norm', action='store_true')
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
        # Loss
        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                            help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
        parser.add_argument('--sample_radius', default=1.5, type=float)
        parser.add_argument("--lw_reg", type=float, default=0.2,
                            help="weight for span loss, set to 0 will ignore")
        parser.add_argument("--lw_cls", type=float, default=1.,
                            help="weight for span loss, set to 0 will ignore")
        parser.add_argument("--lw_sal", type=float, default=0.1,
                            help="weight for span loss, set to 0 will ignore")
        parser.add_argument("--lw_saliency", type=float, default=0.1,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_wattn", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_ms_align", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument('--span_loss_coef', default=10, type=float)
        parser.add_argument('--giou_loss_coef', default=3, type=float)
        parser.add_argument('--label_loss_coef', default=4, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        # Optional existence head (GMR): predict whether query-video pair contains any moment.
        parser.add_argument("--use_exist_head", action="store_true",
                            help="Enable existence head for positive/negative (GMR) prediction.")
        parser.add_argument("--exist_pool", type=str, default="mean", choices=["mean", "max"],
                            help="Video global pooling type for existence head.")
        parser.add_argument("--exist_loss_coef", type=float, default=1.0,
                            help="Weight for existence BCE loss.")
        parser.add_argument("--exist_gate_thd", type=float, default=0.5,
                            help="Soft gate threshold used in inference; if p_exist<thd, score*=p_exist.")
        parser.add_argument("--pred_topk_for_cls", type=int, default=10,
                            help="Top-K windows used for GMR positive/negative classification metrics.")
        parser.add_argument("--pred_score_thd_for_cls", type=float, default=0.5,
                            help="Score threshold used to classify a query-video pair as positive in GMR metrics.")
        # Part 2 parameters
        parser.add_argument(
            "--variant", type=str, default=None,
            choices=["G0-Threshold", "G0", "G0-Con", "P0", "P0-R", "P0-AllK", "C1", "C2"],
            help="Locked Part 2 variant name",
        )
        parser.add_argument("--baseline_index", type=str, default=None, help="Baseline index file path")
        parser.add_argument("--enable_adapter", action="store_true", help="Enable P0 event adapter")
        parser.add_argument(
            "--adapter_variant", type=str, default="P0",
            choices=["P0", "P0-R", "P0-AllK"], help="P0 adapter variant",
        )
        parser.add_argument("--enable_aec", action="store_true", help="Enable event cardinality (AEC)")
        parser.add_argument("--aec_variant", type=str, default="C1", choices=["G0", "G0-Con", "C1", "C2"], help="AEC variant")
        parser.add_argument("--init_backbone_ckpt", type=str, default=None, help="Backbone checkpoint to initialize from")
        parser.add_argument("--adapter_ckpt", type=str, default=None, help="Adapter checkpoint to load for AEC training")
        parser.add_argument("--freeze_adapter", action="store_true", help="Freeze adapter weights during AEC training")
        parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")
        parser.add_argument("--count_calibration", type=str, default=None, help="Count calibration JSON file")

        parser.add_argument("--no_sort_results", action="store_true",
                            help="do not sort results, use this for moment query visualization")
        parser.add_argument("--max_before_nms", type=int, default=50)
        parser.add_argument("--max_after_nms", type=int, default=10)
        parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
        parser.add_argument("--nms_thd", type=float, default=0.7,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")
        parser.add_argument("--nms_type", type=str, default="normal", choices=["normal", "linear"])
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self, a_feat_dir=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # if opt.debug:
        #     opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
        #     opt.num_workers = 0

        if isinstance(self, TestOptions):
            # Preserve CLI overrides before loading saved opt.json.
            _cli_overrides = {
                "device": getattr(opt, "device", None),
                "v_feat_dirs": getattr(opt, "v_feat_dirs", None),
                "t_feat_dir": getattr(opt, "t_feat_dir", None),
                "v_feat_dim": getattr(opt, "v_feat_dim", None),
                "t_feat_dim": getattr(opt, "t_feat_dim", None),
                "feature_manifest": getattr(opt, "feature_manifest", None),
                "data_manifest_index": getattr(opt, "data_manifest_index", None),
                "baseline_index": getattr(opt, "baseline_index", None),
                "count_calibration": getattr(opt, "count_calibration", None),
            }
            requested_variant = getattr(opt, "variant", None)

            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            # if a_feat_dir is not None:
            #     opt.a_feat_dir = a_feat_dir
            saved_opt_path = getattr(opt, "opt_path", None) or os.path.join(opt.model_dir, self.saved_option_filename)
            saved_options = load_json(saved_opt_path)
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",
                               "eval_split_name", "eval_path",
                               "max_pred_l", "min_pred_l",
                               "resume", "resume_all", "no_sort_results"]:
                    setattr(opt, arg, saved_options[arg])
            saved_variant = getattr(opt, "variant", None)
            # G0-Threshold is the one calibration-only Part 2 variant: it
            # intentionally reuses the indexed B0 checkpoint and therefore has
            # no same-variant training opt.json of its own.
            is_threshold_from_b0 = requested_variant == "G0-Threshold" and saved_variant is None
            if requested_variant is not None and requested_variant != saved_variant and not is_threshold_from_b0:
                raise ValueError(
                    f"Inference variant override is forbidden: checkpoint={saved_variant}, CLI={requested_variant}"
                )
            if is_threshold_from_b0:
                opt.variant = requested_variant
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
                mkdirp(opt.results_dir)

            # Re-apply CLI overrides (if provided)
            for k, v in _cli_overrides.items():
                if v is not None:
                    setattr(opt, k, v)
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            if getattr(opt, "resume", None) is not None:
                opt.results_dir = os.path.dirname(os.path.abspath(opt.resume))
            elif getattr(opt, "variant", None) is not None:
                # Part 2 run directories are explicit artifact identities supplied
                # by the orchestration script; do not hide them under a timestamp.
                opt.results_dir = os.path.abspath(opt.results_root)
            else:
                ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
                opt.results_dir = os.path.join(opt.results_root,
                                               "-".join([opt.dset_name, ctx_str, opt.exp_id, time.strftime("%Y-%m-%d-%H-%M-%S")]))

                                                    #  str(opt.enc_layers) + str(opt.dec_layers) + str(opt.t2v_layers) + str(opt.moment_layers) + str(opt.dummy_layers) + str(opt.sent_layers),
                                                    #  'ndum_' + str(opt.num_dummies), 'nprom_' + str(opt.num_prompts) + '_' + str(opt.total_prompts)]))

            mkdirp(opt.results_dir)
            # Save the training package used for this run.
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        if getattr(opt, "strict_data_contract", False):
            if opt.baseline_variant != "B0-gt-only":
                opt.require_text_mask = True
            if opt.baseline_variant == "B0-gt-only":
                opt.legacy_text_mask = True
            if opt.txt_drop_ratio != 0:
                raise ValueError("Strict B0 requires --txt_drop_ratio 0")
            if opt.eval_bsz != 1:
                raise ValueError(
                    "Strict B0 requires --eval_bsz 1 because boundary decoding is single-sample"
                )
            if opt.baseline_variant in {"B0-gt-only", "B0"} and opt.max_windows != -1:
                raise ValueError(f"{opt.baseline_variant} requires --max_windows -1")
            if opt.feature_manifest is None:
                raise ValueError("Strict B0 requires --feature_manifest")
            self._validate_feature_manifest(opt)
            if opt.data_manifest_index is None:
                raise ValueError("Strict B0 requires --data_manifest_index")
            self._validate_data_manifest(opt)
        if getattr(opt, "variant", None) is not None:
            if opt.max_windows != -1:
                raise ValueError("Part 2 requires --max_windows -1")
            if not opt.strict_data_contract:
                raise ValueError("Part 2 requires --strict_data_contract")
            if opt.baseline_index is None:
                raise ValueError("Part 2 requires --baseline_index")
            if opt.variant in {"G0-Threshold", "G0", "G0-Con", "P0", "P0-R", "P0-AllK"} and opt.init_backbone_ckpt is None and opt.resume is None:
                raise ValueError(f"{opt.variant} requires --init_backbone_ckpt")
            if opt.variant in {"C1", "C2"}:
                if opt.adapter_ckpt is None and opt.resume is None:
                    raise ValueError(f"{opt.variant} requires --adapter_ckpt")
                if not opt.freeze_adapter:
                    raise ValueError(f"{opt.variant} requires --freeze_adapter")
        if getattr(opt, "repro_check", False):
            opt.num_workers = 0

        self.display_save(opt)
        if not isinstance(self, TestOptions):
            with open(os.path.join(opt.results_dir, "command.txt"), "w", encoding="utf-8") as handle:
                handle.write(" ".join([sys.executable, "-m", "training.flash_vtg_gmr.train", *sys.argv[1:]]) + "\n")
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
            }
            save_json(
                environment,
                os.path.join(opt.results_dir, "environment.txt"),
                save_pretty=True,
            )

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device(f"cuda:{opt.device}" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt

    @staticmethod
    def _validate_feature_manifest(opt):
        with open(opt.feature_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        stored_sha = manifest.get("content_sha256")
        payload = dict(manifest)
        payload.pop("content_sha256", None)
        if stored_sha != canonical_json_sha256(payload):
            raise ValueError("Feature manifest content SHA256 mismatch")
        if manifest.get("concat_order") != ["slowfast", "clip"]:
            raise ValueError("Feature manifest requires [slowfast, clip] order")
        if manifest.get("setting") != "f-lighthouse":
            raise ValueError(
                f"Formal Part 1 runs require setting=f-lighthouse, got {manifest.get('setting')}"
            )
        if manifest.get("stream_dimensions") != {"slowfast": 2304, "clip": 512}:
            raise ValueError("Feature manifest stream dimensions mismatch")
        if int(manifest.get("video_feature_dim", -1)) != 2816:
            raise ValueError("Feature manifest video dimension must be 2816")
        if int(manifest.get("text_feature_dim", -1)) != 512:
            raise ValueError("Feature manifest text dimension must be 512")
        if manifest.get("per_stream_normalization") is not True:
            raise ValueError("Formal Part 1 runs require per-stream normalization")
        if float(manifest.get("normalization_eps", -1)) != 1e-5:
            raise ValueError("Formal Part 1 runs require normalization_eps=1e-5")
        if manifest.get("known_provenance") is not True:
            raise ValueError("Formal Part 1 runs require known Lighthouse provenance")
        if manifest.get("encoder_mode") != "eval":
            raise ValueError("Formal Part 1 runs require Lighthouse encoders in eval mode")
        if manifest.get("text_token_alignment_status") != "verified":
            raise ValueError("Formal Part 1 runs require verified text token alignment")
        if manifest.get("cross_stream_length_policy") != (
            "lighthouse-trim-shorter-at-extraction; exact-or-fail-loader"
        ):
            raise ValueError("Unexpected Lighthouse cross-stream length policy")
        batch_audit = manifest.get("batch_invariance_audit", {})
        if int(batch_audit.get("shared_video_count", 0)) < 50:
            raise ValueError("Formal Part 1 runs require a 50-video batch-invariance audit")

        def _verify_artifact(path, expected_sha, label):
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Missing frozen {label}: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Frozen {label} SHA256 mismatch: {actual_sha} != {expected_sha}"
                )

        for key in (
            "extraction_provenance",
            "numerical_audit",
            "identity_audit",
            "text_alignment_audit",
            "batch_invariance_audit",
        ):
            record = manifest.get(key, {})
            _verify_artifact(record.get("path"), record.get("sha256"), key)
        runtime_records = manifest.get("extraction_runtime", {})
        if set(runtime_records) != {"video_shard_0", "video_shard_1", "text_shard_0"}:
            raise ValueError("Feature manifest extraction runtime inventory mismatch")
        for key, record in runtime_records.items():
            _verify_artifact(record.get("path"), record.get("sha256"), key)
        for record in manifest.get("video_inventory", {}).values():
            _verify_artifact(
                record.get("slowfast_path"),
                record.get("slowfast_sha256"),
                "SlowFast feature",
            )
            _verify_artifact(
                record.get("clip_path"), record.get("clip_sha256"), "CLIP feature"
            )
        for record in manifest.get("query_inventory", {}).values():
            _verify_artifact(
                record.get("text_path"), record.get("text_sha256"), "text feature"
            )
        expected_dirs = [manifest.get("slowfast_dir"), manifest.get("clip_dir")]
        actual_dirs = list(opt.v_feat_dirs or [])
        if len(actual_dirs) != 2:
            raise ValueError("Strict feature contract requires exactly two video feature directories")
        resolved_expected = [os.path.realpath(path) for path in expected_dirs]
        resolved_actual = [os.path.realpath(path) for path in actual_dirs]
        if resolved_expected != resolved_actual:
            raise ValueError(
                f"Video feature order/path mismatch: expected {expected_dirs}, got {actual_dirs}"
            )
        if os.path.realpath(manifest.get("text_dir")) != os.path.realpath(opt.t_feat_dir):
            raise ValueError("Text feature directory does not match feature manifest")
        if float(manifest.get("clip_length")) != float(opt.clip_length):
            raise ValueError("clip_length does not match feature manifest")
        if int(manifest.get("text_context_length_model")) != int(opt.max_q_l):
            raise ValueError("max_q_l does not match feature manifest")
        if int(opt.v_feat_dim) != int(manifest["video_feature_dim"]):
            raise ValueError("v_feat_dim does not match feature manifest")
        if int(opt.t_feat_dim) != int(manifest["text_feature_dim"]):
            raise ValueError("t_feat_dim does not match feature manifest")
        if int(opt.max_v_l) != 75:
            raise ValueError("Strict B0 requires max_v_l=75")
        if opt.ctx_mode != "video_tef":
            raise ValueError("Strict B0 requires ctx_mode=video_tef")
        if opt.no_norm_vfeat or opt.no_norm_tfeat:
            raise ValueError("Strict B0 requires video and text normalization")

    @staticmethod
    def _validate_data_manifest(opt):
        with open(opt.data_manifest_index, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        stored_sha = index.get("content_sha256")
        payload = dict(index)
        payload.pop("content_sha256", None)
        if stored_sha != canonical_json_sha256(payload):
            raise ValueError("Data manifest index content SHA256 mismatch")
        with open(opt.feature_manifest, "r", encoding="utf-8") as handle:
            feature_manifest = json.load(handle)
        if index.get("dataset_setting") != feature_manifest.get("dataset_setting"):
            raise ValueError("Data/feature manifest dataset_setting mismatch")
        if index["feature_manifest"]["content_sha256"] != feature_manifest["content_sha256"]:
            raise ValueError("Data manifest was built from a different feature manifest")

        paths_to_validate = []
        if opt.train_path is not None:
            paths_to_validate.append(("train", opt.train_path))
        if opt.eval_path is not None:
            paths_to_validate.append((str(opt.eval_split_name), opt.eval_path))
        for split, actual_path in paths_to_validate:
            if split not in index["data_manifests"]:
                raise ValueError(f"Data manifest index has no split {split!r}")
            record = index["data_manifests"][split]
            if os.path.realpath(record["path"]) != os.path.realpath(actual_path):
                raise ValueError(
                    f"{split} path does not match canonical manifest: {actual_path}"
                )
            if sha256_file(actual_path) != record["sha256"]:
                raise ValueError(f"Canonical {split} manifest SHA256 mismatch")


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument(
            "--opt_path",
            type=str,
            default=None,
            help="Path to opt.json to load (override default: dirname(--resume)/opt.json).",
        )
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
