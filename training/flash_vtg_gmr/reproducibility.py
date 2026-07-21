import hashlib
import random

import numpy as np
import torch


def seed_everything(seed, use_cuda=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(repro_check=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(repro_check)
    torch.use_deterministic_algorithms(bool(repro_check))


def make_data_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state(data_generator=None, dataset_epoch=None, global_step=None):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataset_epoch": dataset_epoch,
        "global_step": global_step,
    }
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def restore_rng_state(state, data_generator=None):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_generator is not None and state.get("data_generator") is not None:
        data_generator.set_state(state["data_generator"])


def tensor_state_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
