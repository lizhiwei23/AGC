import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random
import time

import clip
import foolbox as fb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset import (
    CLIPClassifierWrapper,
    CustomDataset,
    EvalDataset,
    collate,
    load_datasets_with_split,
    mapping,
)
from utils.tta import Adaptive_geo_cor
from utils.tools import evaluate_with_detailed_print, load_or_generate_adv


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clip_normalize_batch(images, device=None):
    if device is None:
        device = images.device
    mean = CLIP_MEAN.to(device=device, dtype=images.dtype)
    std = CLIP_STD.to(device=device, dtype=images.dtype)
    return (images - mean) / std


@torch.no_grad()
def encode_clip_images(model, images):    
    images = clip_normalize_batch(images)
    feats = model.encode_image(images).float()
    return F.normalize(feats, dim=-1)



if __name__ == "__main__":
    print("Usage: python test.py [options]")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./datasets/caltech-101")
    parser.add_argument("--model_name", default="ViT-B/32")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="rtpt",
        choices=["default", "rtpt"],
        help="text prompt style: default='a photo of a {class}', rtpt='a photo of a {class}.' with '_'->' '",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.add_argument("--epsilon", type=float, default=4 / 255)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rel_stepsize", type=float, default=0.25)
    parser.add_argument("--random_start", action="store_true", default=True)
    parser.add_argument("--no_random_start", action="store_false", dest="random_start")
    parser.add_argument("--attack_bs", type=int, default=64)
    parser.add_argument("--adv_cache_dir", default=None)
    parser.add_argument("--n_views", type=int, default=32)
    parser.add_argument("--datatype", type=str, default="caltech101")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
	
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Seed: {args.seed} | Deterministic: {args.deterministic}")
  
    if args.adv_cache_dir is None:
        eps_tag = int(round(args.epsilon * 255))
        args.adv_cache_dir = (
            f"./adv_cache/{args.datatype}_eps{eps_tag}_steps{args.steps}_{args.model_name.replace('/', '-')}"
            f"_rs{int(args.random_start)}"
        )

    cfg = {
        "dataset_root": args.dataset_root,
        "datatype": args.datatype,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples,
        "n_views": args.n_views,
    }

    script_start_time = time.perf_counter()
    device = args.device
    print(f"Device: {device}")

    print("Loading dataset...")
    print(f"  Dataset root : {cfg['dataset_root']}")
    print(f"  Split file   : {os.path.join(cfg['dataset_root'], mapping[cfg['datatype']][1])}")

    clean_samples, class_names = load_datasets_with_split(
        cfg["dataset_root"], type=cfg["datatype"], split="test"
    )
    print(f"\nTest size: {len(clean_samples)}")
    print(f"Class names ({len(class_names)}): {class_names[:5]} ...")

    print("Loading CLIP model...")
    model, _ = clip.load(cfg["model_name"], device=device)
    model.eval()

    if args.prompt_style == "rtpt":
        prompt_class_names = [c.replace("_", " ") for c in class_names]
        prompt_template = "a photo of a {}."
    else:
        prompt_class_names = class_names
        prompt_template = "a photo of a {}"

    prompts = [prompt_template.format(c) for c in prompt_class_names]
    print(f"Prompt style: {args.prompt_style}")
    print(f"Prompt example: {prompts[0]}")

    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)

    wrapped = CLIPClassifierWrapper(model, text_features).to(device).eval()
    fmodel = fb.PyTorchModel(wrapped, bounds=(0, 1), device=device)

    print("\nPreparing adversarial samples...")
    adv_samples = load_or_generate_adv(
        clean_path_label_list=clean_samples,
        fmodel=fmodel,
        epsilon=args.epsilon,
        steps=args.steps,
        rel_stepsize=args.rel_stepsize,
        random_start=args.random_start,
        adv_cache_dir=args.adv_cache_dir,
        device=device,
        max_samples=cfg["max_samples"],
        attack_bs=args.attack_bs,
        model_name=cfg["model_name"],
    )

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    n = len(adv_samples)
    clean_eval = CustomDataset(clean_samples[:n], preprocess, type=cfg["datatype"])
    adv_eval = EvalDataset(adv_samples, type=cfg["datatype"], transform=preprocess)

    print(f"\nClean dataset size : {len(clean_eval)}")
    print(f"Adv   dataset size : {len(adv_eval)}")

    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )
    clean_loader = DataLoader(clean_eval, **loader_kwargs)
    adv_loader = DataLoader(adv_eval, **loader_kwargs)

    defense_label = "Adaptive_geo_cor"
    print(f"Initializing Defense ({defense_label})...")
    defense = Adaptive_geo_cor(
        clip_model=model,
        all_classes=class_names,
        text_features=text_features,
        prompt_style=args.prompt_style,
        n_views=cfg["n_views"],
        device=device,
    )

    print("Starting Evaluation...\n")
    evaluate_with_detailed_print(
        model,
        text_features,
        defense,
        clean_loader,
        adv_loader,
        class_names,
        device,
        max_samples=cfg["max_samples"],
    )

    script_end_time = time.perf_counter()
    print(f"Total Script Runtime: {script_end_time - script_start_time:.2f} seconds")
