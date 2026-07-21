import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.flash_vtg_gmr.config import BaseOptions
from training.flash_vtg_gmr.dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from training.flash_vtg_gmr.inference import eval_epoch, start_inference, setup_model
from models.flash_vtg_gmr.utils.basic_utils import AverageMeter, dict_to_markdown

import nncore
from datetime import datetime
import logging
from training.flash_vtg_gmr.reproducibility import (
    capture_rng_state,
    configure_runtime,
    make_data_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)
from training.flash_vtg_gmr.contracts import sha256_file


def _selection_metric_names(variant):
    if variant in {"P0", "P0-R"}:
        return ["AdapterScore"]
    if variant in {"G0", "G0-Con", "C1", "C2"}:
        return ["SetSuccess@0.5", "MR-full-mAP", "Count-Acc-5"]
    return ["MR-full-mAP"]


def _training_command(opt):
    path = os.path.join(opt.results_dir, "command.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _strict_checkpoint_payload(
    model, optimizer, lr_scheduler, epoch_i, opt, data_generator, iteration,
    prev_best_key, es_cnt, validation_files=None,
):
    validation_files = validation_files or []
    validation_records = {
        "predictions": None,
        "metrics": None,
    }
    if len(validation_files) >= 2:
        validation_records = {
            "predictions": {
                "path": validation_files[0],
                "sha256": sha256_file(validation_files[0]),
            },
            "metrics": {
                "path": validation_files[1],
                "sha256": sha256_file(validation_files[1]),
            },
        }
    variant = getattr(opt, "variant", None)
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch_i,
        "opt": opt,
        "seed": int(opt.seed),
        "variant": variant,
        "baseline_checkpoint_sha256": getattr(opt, "baseline_checkpoint_sha256", None),
        "feature_manifest_sha256": getattr(opt, "feature_manifest_sha256", None),
        "public_p0_checkpoint_sha256": getattr(opt, "public_p0_checkpoint_sha256", None),
        "event_interface_schema": "EventInterfaceV1" if getattr(opt, "enable_adapter", False) else None,
        "event_interface_metadata": {
            "schema": "EventInterfaceV1",
            "span_source": "greedy_seed_spans",
            "selection_threshold": 0.5,
            "max_modes": int(getattr(opt, "event_max_modes", 10)),
        } if variant in {"P0", "P0-R"} else None,
        "checkpoint_boundary": "epoch",
        "selection_split": "val",
        "selection_metric_names": _selection_metric_names(variant),
        "selection_key": list(prev_best_key),
        "validation_artifacts": validation_records,
        "training_command": _training_command(opt),
        "reproducibility_state": capture_rng_state(
            data_generator, dataset_epoch=epoch_i, global_step=iteration
        ),
        "training_state": {
            "prev_best_key": list(prev_best_key),
            "es_cnt": es_cnt,
        },
    }

def set_seed(seed, use_cuda=True):
    seed_everything(seed, use_cuda=use_cuda)

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)

    # iteration loop
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(
        enumerate(train_loader), desc="Training Iteration", total=num_training_examples
    ):


        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        targets["label"] = batch[0]
        bsz = int(model_inputs["src_vid"].shape[0])
        targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)  # fps=1/clip_len

        # Compute count class for AEC
        counts = []
        for meta in batch[0]:
            rel_wins = meta.get("relevant_windows", [])
            if rel_wins is None:
                rel_wins = []
            counts.append(min(len(rel_wins), 4))
        targets["count_class"] = torch.tensor(counts, dtype=torch.long, device=opt.device)

        outputs = model(**model_inputs, targets=targets)

        loss_dict = criterion(batch, outputs, targets)
        # keep only scalar loss entries
        loss_dict = {k: v for k, v in loss_dict.items() if "loss" in k}

        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        nonfinite = {
            key: value
            for key, value in loss_dict.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        }
        if nonfinite or not torch.isfinite(losses).all():
            identities = [(meta.get("qid"), meta.get("vid")) for meta in batch[0]]
            raise FloatingPointError(
                f"Non-finite loss for identities={identities}, terms={list(nonfinite)}"
            )

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), opt.grad_clip, error_if_nonfinite=True
            )
        optimizer.step()

        if opt.repro_check:
            trace_path = os.path.join(opt.results_dir, "repro_trace.jsonl")
            trace_record = {
                "epoch": int(epoch_i),
                "step": int(batch_idx),
                "qids": [meta.get("qid") for meta in batch[0]],
                "weighted_loss": float(losses.detach()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            with open(trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_record, sort_keys=True) + "\n")

        loss_dict["weighted_loss_overall"] = float(losses.detach())  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v.detach()) if torch.is_tensor(v) else float(v)
            )

        # Output and log loss info every iteration
        current_loss = {k: v.avg for k, v in loss_meters.items()}
        for k, v in current_loss.items():
            tb_writer.add_scalar(f"Train/{k}", v, epoch_i * num_training_examples + batch_idx)

        tb_writer.add_scalar(
            "Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i * num_training_examples + batch_idx
        )

        if 0 < opt.max_train_steps <= batch_idx + 1:
            break

    # Write epoch-level logs to file
    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i + 1,
        loss_str=" ".join(
            ["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]
        ),
    )
    logger.info(to_write)
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    return losses, epoch_i * num_training_examples + batch_idx

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    data_generator = make_data_generator(opt.seed)
    resume_rng_state = getattr(opt, "_resume_rng_state", None)
    if resume_rng_state:
        restore_rng_state(resume_rng_state, data_generator=data_generator)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
        drop_last=opt.drop_last,
        generator=data_generator,
        worker_init_fn=seed_worker,
    )

    resume_training_state = getattr(opt, "_resume_training_state", None) or {}
    stored_best_key = resume_training_state.get("prev_best_key")
    if stored_best_key is not None:
        prev_best_key = tuple(float(value) for value in stored_best_key)
    else:
        prev_best_key = (float(resume_training_state.get("prev_best_score", -1e9)),)
    es_cnt = int(resume_training_state.get("es_cnt", 0))
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    iteration = max(start_epoch, 0) * len(train_loader)
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name
    )

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        train_dataset.set_epoch(epoch_i)
        if epoch_i > -1:
            losses, iteration = train_epoch(
                model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer
            )
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch

        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        save_submission_filename,
                        epoch_i,
                        criterion,
                        tb_writer,
                    )
                )

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(
                    ["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]
                ),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info(
                "metrics_no_nms {}".format(
                    pprint.pformat(metrics_no_nms["brief"], indent=4)
                )
            )
            if metrics_nms is not None:
                logger.info(
                    "metrics_nms {}".format(
                        pprint.pformat(metrics_nms["brief"], indent=4)
                    )
                )

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), iteration)
            if "GMR-TPR" in metrics["brief"]:
                logger.info(
                    "GMR Metrics - TPR: {:.2f}, TNR: {:.2f}, BalancedAcc: {:.2f}".format(
                        metrics["brief"]["GMR-TPR"],
                        metrics["brief"]["GMR-TNR"],
                        metrics["brief"]["GMR-BalancedAcc"],
                    )
                )

            if getattr(opt, "variant", None) in {"P0", "P0-R"}:
                stop_key = (float(metrics["brief"]["AdapterScore"]),)
            elif getattr(opt, "variant", None) in {"G0", "G0-Con", "C1", "C2"}:
                stop_key = (
                    float(metrics["brief"]["SetSuccess@0.5"]),
                    float(metrics["brief"]["MR-full-mAP"]),
                    float(metrics["brief"]["Count-Acc-5"]),
                )
            elif opt.dset_name in ["hl"]:
                stop_key = (float(metrics["brief"]["MR-full-mAP"]),)
            elif opt.dset_name in ["tacos"]:
                stop_key = (float(metrics["brief"]["MR-full-R1@0.3"]),)
            else:
                stop_key = ((
                    metrics["brief"]["MR-full-R1@0.7"]
                    + metrics["brief"]["MR-full-R1@0.5"]
                ) / 2,)

            if stop_key > prev_best_key:
                es_cnt = 0
                prev_best_key = stop_key
                best_file_paths = [
                    e.replace("latest", "best") for e in latest_file_paths
                ]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                checkpoint = _strict_checkpoint_payload(
                    model, optimizer, lr_scheduler, epoch_i, opt, data_generator,
                    iteration, prev_best_key, es_cnt, best_file_paths,
                )
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(
                        f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_key}\n"
                    )
                    break

        # save ckpt
        checkpoint = _strict_checkpoint_payload(
            model, optimizer, lr_scheduler, epoch_i, opt, data_generator,
            iteration, prev_best_key, es_cnt,
        )
        torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()

def train_hl(
    model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt
):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
        drop_last=opt.drop_last,
    )

    prev_best_score = 0.0
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(
        opt.dset_name, opt.eval_split_name
    )
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(
                model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer
            )
            lr_scheduler.step() # use step() for StepLR not ReduceLROnPlateau
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        save_submission_filename,
                        epoch_i,
                        criterion,
                        tb_writer,
                    )
                )

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(
                    ["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]
                ),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info(
                "metrics_no_nms {}".format(
                    pprint.pformat(metrics_no_nms["brief"], indent=4)
                )
            )
            if metrics_nms is not None:
                logger.info(
                    "metrics_nms {}".format(
                        pprint.pformat(metrics_nms["brief"], indent=4)
                    )
                )

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i + 1)
            if "GMR-TPR" in metrics["brief"]:
                logger.info(
                    "GMR Metrics - TPR: {:.2f}, TNR: {:.2f}, BalancedAcc: {:.2f}".format(
                        metrics["brief"]["GMR-TPR"],
                        metrics["brief"]["GMR-TNR"],
                        metrics["brief"]["GMR-BalancedAcc"],
                    )
                )

            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [
                    e.replace("latest", "best") for e in latest_file_paths
                ]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(
                        f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n"
                    )
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt,
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()

def start_training():
    logger.info("Setup data and model...")

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
        mr_only=opt.mr_only,
        keep_empty_gt=(getattr(opt, "variant", None) is not None or bool(getattr(opt, "use_exist_head", False))),
        strict_data_contract=getattr(opt, "strict_data_contract", False),
        require_text_mask=getattr(opt, "require_text_mask", False),
        text_store_length=getattr(opt, "text_store_length", 77),
        legacy_text_mask=getattr(opt, "legacy_text_mask", False),
        legacy_gt_sampling=getattr(opt, "legacy_gt_sampling", False),
        seed=opt.seed,
    )
    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss

        eval_dataset = StartEndDataset(**dataset_config)

    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)

    if getattr(opt, "enable_aec", False):
        from collections import Counter
        from models.flash_vtg_gmr.event_cardinality import compute_effective_number_weights
        counts = Counter()
        for item in train_dataset.data:
            rel_wins = item.get("relevant_windows", [])
            if rel_wins is None:
                rel_wins = []
            counts[min(len(rel_wins), 4)] += 1
        
        weights = compute_effective_number_weights(counts)
        logger.info(f"AEC Class weights: {weights.tolist()}")
        if hasattr(model, "aec") and model.aec is not None:
            model.aec.register_buffer("class_weights", weights.to(opt.device))
    logger.info(f"Model {model}")
    params = []
    logger.info("Learnable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.info(f"{name} - {param.shape}")
            params.append(param)

    train_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = round(train_params / total_params * 100, 3)
    param = round(train_params / 1024 / 1024, 3)
    logger.info(f"Learnable Parameters: {param}M ({ratio}%)")

    logger.info("Start Training...")

    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum', 'youtube_uni']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    return (
        opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"),
        opt.eval_split_name,
        opt.eval_path,
        opt.debug,
        opt,
    )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    configure_runtime(repro_check=bool(opt.repro_check or opt.debug))

    opt.cfg = nncore.Config.from_file(opt.config)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_directory = os.path.join(opt.results_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
    file_handler = logging.FileHandler(log_directory)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()

    # Count-based Part 2 variants cannot run their formal inference until the
    # validation calibration artifact has been fitted.  Their validation
    # predictions/metrics are already emitted by the epoch evaluator and are
    # subsequently consumed by finalize_part2_run.sh, which calibrates first
    # and only then runs inference plus exact replay.
    needs_count_calibration = getattr(opt, "variant", None) in {
        "G0",
        "G0-Con",
        "C1",
        "C2",
    }
    if not debug and eval_path is not None and not needs_count_calibration:
        input_args = [
            opt.config,
            "--resume",
            best_ckpt_path,
            "--eval_split_name",
            eval_split_name,
            "--eval_path",
            eval_path,
        ]

        import sys

        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
    elif needs_count_calibration:
        logger.info(
            "Skipping post-training inference for %s; calibration and formal "
            "inference are performed by finalize_part2_run.sh",
            opt.variant,
        )
