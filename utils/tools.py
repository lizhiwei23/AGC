import time
from torchvision import transforms
import json
import os
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


def format_running_summary(results, processed, defense_name):
    if processed <= 0:
        return f"[{defense_name}] no samples processed yet"

    clean_acc = results['clean_correct'] / processed * 100
    adv_acc = results['adv_correct'] / processed * 100
    tta_clean_acc = results['tta_clean_correct'] / processed * 100
    tta_adv_acc = results['tta_adv_correct'] / processed * 100

    return (
        f"[{defense_name}] {processed} samples | "
        f"clean={clean_acc:.2f}% adv={adv_acc:.2f}% "
        f"def_clean={tta_clean_acc:.2f}% def_adv={tta_adv_acc:.2f}% "
    )


def evaluate_with_detailed_print(
    model, text_features, defense, clean_loader, adv_loader, 
    class_names, device, max_samples=-1
):
    defense_name = getattr(defense, "defense_name", "Defense")
    
    results = {
        'clean_correct': 0,
        'adv_correct': 0,
        'tta_clean_correct': 0,
        'tta_adv_correct': 0,
        'attack_success': 0,
        'total': 0,
    }

    timer_total_start = time.perf_counter()
    
    sample_count = 0

    dataset_total = None
    if hasattr(clean_loader, "dataset") and hasattr(adv_loader, "dataset"):
        dataset_total = min(len(clean_loader.dataset), len(adv_loader.dataset))
    if max_samples > 0:
        total_for_bar = min(dataset_total, max_samples) if dataset_total is not None else max_samples
    else:
        total_for_bar = dataset_total

    count = 0
    text_features_t = text_features.T
    progress = tqdm(total=total_for_bar, desc=f"Eval {defense_name}", dynamic_ncols=True)
    try:
        clean_iter = iter(clean_loader)
        adv_iter = iter(adv_loader)
        while True:
            if max_samples > 0 and count >= max_samples:
                break

            try:
                clean_batch = next(clean_iter)
                adv_batch = next(adv_iter)
            except StopIteration:
                break

            clean_images = clean_batch["image"].to(device)
            clean_labels = clean_batch["label"].to(device)
            clean_pils = clean_batch["pil"]

            adv_images = adv_batch["image"].to(device)
            adv_pils = adv_batch["pil"]

            batch_n = len(clean_labels)
            if max_samples > 0:
                remaining = max_samples - count
                if remaining <= 0:
                    break
                if remaining < batch_n:
                    batch_n = remaining
                    clean_images = clean_images[:batch_n]
                    clean_labels = clean_labels[:batch_n]
                    clean_pils = clean_pils[:batch_n]
                    adv_images = adv_images[:batch_n]
                    adv_pils = adv_pils[:batch_n]

            with torch.no_grad():
                clean_feat = F.normalize(model.encode_image(clean_images).float(), dim=-1)
                clean_logits = 100.0 * clean_feat @ text_features_t
                clean_preds = clean_logits.argmax(dim=1)

                adv_feat = F.normalize(model.encode_image(adv_images).float(), dim=-1)
                adv_logits = 100.0 * adv_feat @ text_features_t
                adv_preds = adv_logits.argmax(dim=1)

            for i in range(batch_n):
                true_idx = clean_labels[i].item()
                clean_pred = int(clean_preds[i].item())
                adv_pred = int(adv_preds[i].item())

                tta_clean_pred = defense.predict_pil(clean_pils[i])

                tta_adv_pred = defense.predict_pil(adv_pils[i])

                clean_ok = clean_pred == true_idx
                adv_ok = adv_pred == true_idx
                tta_clean_ok = tta_clean_pred == true_idx
                tta_adv_ok = tta_adv_pred == true_idx


                results['clean_correct'] += int(clean_ok)
                results['adv_correct'] += int(adv_ok)
                results['tta_clean_correct'] += int(tta_clean_ok)
                results['tta_adv_correct'] += int(tta_adv_ok)
                results['total'] += 1

                if clean_ok and not adv_ok:
                    results['attack_success'] += 1

                count += 1
                sample_count += 1
                progress.update(1)

                if count % 200 == 0:
                    progress.write(format_running_summary(results, count, defense_name))

                if count % 20 == 0:
                    progress.set_postfix({
                        "clean": f"{results['clean_correct'] / count * 100:.1f}%",
                        "df_c": f"{results['tta_clean_correct'] / count * 100:.1f}%",
                        "df_ad": f"{results['tta_adv_correct'] / count * 100:.1f}%",
                    })
    finally:
        progress.close()

    timer_total_end = time.perf_counter()
    total_elapsed = timer_total_end - timer_total_start

    print(f"{'=' * 120}")
    
    total = results['total']
    if total == 0:
        print(" No samples evaluated.")
        return results

    clean_acc = results['clean_correct'] / total * 100
    adv_acc = results['adv_correct'] / total * 100
    tta_clean_acc = results['tta_clean_correct'] / total * 100
    tta_adv_acc = results['tta_adv_correct'] / total * 100


    print(f"  CLIP Acc:      {clean_acc:6.2f}%  ({results['clean_correct']}/{total})")
    print(f"  CLIP Rob:        {adv_acc:6.2f}%  ({results['adv_correct']}/{total})")
    print(f"  AGC Acc:  {tta_clean_acc:6.2f}%  ({results['tta_clean_correct']}/{total})")
    print(f"  AGC Rob:    {tta_adv_acc:6.2f}%  ({results['tta_adv_correct']}/{total})")
    
    
    print(f"  Total Evaluation Time:     {total_elapsed:6.2f} s")
    print(f"  Throughput (Samples/s):    {total / total_elapsed:6.2f}")
    
    print(f"{'=' * 120}\n")

    return results

def load_or_generate_adv(
    clean_path_label_list,
    fmodel,
    epsilon,
    steps,
    rel_stepsize,
    random_start,
    adv_cache_dir,
    device,
    max_samples=-1,
    attack_bs=32,
    model_name="ViT-B/32",
):
    n = len(clean_path_label_list) if max_samples <= 0 else min(max_samples, len(clean_path_label_list))
    samples = clean_path_label_list[:n]

    os.makedirs(adv_cache_dir, exist_ok=True)
    meta_path = os.path.join(adv_cache_dir, "meta.json")

    cache_ok = False
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("n") == n
            and abs(meta.get("epsilon", -1) - epsilon) < 1e-9
            and meta.get("steps") == steps
            and meta.get("rel_stepsize") == rel_stepsize
            and meta.get("random_start") == random_start
            and meta.get("model_name") == model_name
            and os.path.isfile(os.path.join(adv_cache_dir, f"{n - 1}.png"))
        ):
            cache_ok = True

    if cache_ok:
        print(f"Loading adv cache: {adv_cache_dir}  ({n} samples)")
        adv = []
        for i in range(n):
            pil = Image.open(os.path.join(adv_cache_dir, f"{i}.png")).convert("RGB")
            adv.append((pil, samples[i][1]))
        return adv

    print(
        f"  Generating LinfPGD adv  ε={epsilon:.5f}  steps={steps}"
        f"  rel_stepsize={rel_stepsize}  random_start={random_start}"
    )
    print(f"      Cache dir: {adv_cache_dir}")

    attack = fb.attacks.LinfPGD(
        steps=steps,
        rel_stepsize=rel_stepsize,
        random_start=random_start,
    )

    to_01 = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    adv_pils = []
    offset = 0

    with tqdm(total=n, desc="PGD generate") as pbar:
        while offset < n:
            prev_offset = offset
            end = min(offset + attack_bs, n)
            batch_paths = [samples[i][0] for i in range(offset, end)]
            batch_labels = [samples[i][1] for i in range(offset, end)]

            xs = torch.stack([
                to_01(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)
            ys = torch.tensor(batch_labels, dtype=torch.long, device=device)

            _, x_adv_list, _ = attack(fmodel, xs, ys, epsilons=[epsilon])
            x_adv = x_adv_list[0].detach().cpu()

            for j in range(x_adv.size(0)):
                pil = transforms.ToPILImage()(x_adv[j].clamp(0, 1))
                global_i = offset + j
                pil.save(os.path.join(adv_cache_dir, f"{global_i}.png"))
                adv_pils.append((pil, batch_labels[j]))

            offset = end
            pbar.update(end - prev_offset)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "n": n,
                "epsilon": epsilon,
                "steps": steps,
                "rel_stepsize": rel_stepsize,
                "random_start": random_start,
                "model_name": model_name,
            },
            f,
            indent=2,
        )

    print(f"Saved {n} adv samples to {adv_cache_dir}")
    return adv_pils
