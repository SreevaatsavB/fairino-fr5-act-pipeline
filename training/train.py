import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import FR5Dataset
from model import ACT, ACTConfig


def get_device(cfg):
    pref = cfg["training"]["device"]
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(cfg):
    d = cfg["dataset"]
    t = cfg["training"]

    # split by episode index to avoid leakage
    tmp = FR5Dataset(d["root"], chunk_size=d["chunk_size"],
                     use_image=d["use_image"], image_size=tuple(d["image_size"]))
    n_total = int(tmp.info["total_episodes"])
    train_eps, val_eps = FR5Dataset.episode_split(n_total, d["val_frac"], t["seed"])

    train_ds = FR5Dataset(d["root"], d["chunk_size"], d["use_image"],
                          tuple(d["image_size"]), episode_indices=train_eps)
    val_ds   = FR5Dataset(d["root"], d["chunk_size"], d["use_image"],
                          tuple(d["image_size"]), episode_indices=val_eps)

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=t["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"train episodes={len(train_eps)}  frames={len(train_ds)}")
    print(f"val   episodes={len(val_eps)}    frames={len(val_ds)}")
    return train_loader, val_loader, train_ds


def build_model(cfg, stats, device):
    m = cfg["model"]
    d = cfg["dataset"]
    model_cfg = ACTConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        latent_dim=m["latent_dim"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        kl_weight=cfg["training"]["kl_weight"],
        temporal_ensemble_coeff=m.get("temporal_ensemble_coeff"),  # from model: section
    )
    model = ACT(model_cfg, stats).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n/1e6:.1f}M")
    return model


def run_epoch(model, loader, optimizer, cfg, device, train=True):
    model.train(train)
    kl_w = cfg["training"]["kl_weight"]
    clip  = cfg["training"]["grad_clip"]
    log_n = cfg["training"]["log_every"]

    total_l1 = total_kl = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            obs   = batch["observation.state"].to(device)
            acts  = batch["action"].to(device)
            pad   = batch["action_is_pad"].to(device)
            img   = batch.get("observation.image")
            if img is not None:
                img = img.to(device)

            l1, kl = model(obs, acts, pad, img)
            loss   = l1 + kl_w * kl

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            total_l1 += l1.item()
            total_kl  += kl.item()

            if train and (step + 1) % log_n == 0:
                print(f"    step {step+1}/{len(loader)}  "
                      f"l1={l1.item():.4f}  kl={kl.item():.4f}")

    return total_l1 / len(loader), total_kl / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = get_device(cfg)
    print(f"device: {device}")

    train_loader, val_loader, train_ds = build_loaders(cfg)
    stats = train_ds.get_stats()
    model = build_model(cfg, stats, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    max_ep   = cfg["training"]["max_epochs"]
    save_n   = cfg["training"]["save_every"]

    for epoch in range(1, max_ep + 1):
        t0 = time.time()

        print(f"\nepoch {epoch}/{max_ep}")
        train_l1, train_kl = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        val_l1,   _        = run_epoch(model, val_loader,   optimizer, cfg, device, train=False)

        elapsed = time.time() - t0
        print(f"  train l1={train_l1:.4f}  kl={train_kl:.4f}  "
              f"val l1={val_l1:.4f}  ({elapsed:.0f}s)")

        is_best = val_l1 < best_val
        if is_best:
            best_val = val_l1

        if epoch % save_n == 0 or is_best:
            ckpt = {
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_l1":         val_l1,
                "config":         cfg,
                "stats":          stats,
            }
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt, path)
            if is_best:
                torch.save(ckpt, ckpt_dir / "best.pt")
                print(f"  ↑ best  saved → {ckpt_dir / 'best.pt'}")
            else:
                print(f"  saved → {path}")

    print(f"\ndone.  best val l1={best_val:.4f}")


if __name__ == "__main__":
    main()
