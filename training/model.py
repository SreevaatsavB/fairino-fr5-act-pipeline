"""
Wraps lerobot's ACTPolicy so train.py and dataset.py need no changes.
Tested against lerobot >= 0.1.0.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.common.policies.act.configuration_act import ACTConfig as _LRConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy


@dataclass
class ACTConfig:
    state_dim:          int   = 6
    action_dim:         int   = 7
    latent_dim:         int   = 32
    d_model:            int   = 512
    nhead:              int   = 8
    num_encoder_layers: int   = 4
    num_decoder_layers: int   = 7
    dim_feedforward:    int   = 3200
    dropout:            float = 0.1
    chunk_size:         int   = 100
    use_image:          bool  = True
    kl_weight:          float = 10.0


def _lerobot_config(cfg: ACTConfig) -> _LRConfig:
    input_shapes = {"observation.state": [cfg.state_dim]}
    input_norms  = {"observation.state": "mean_std"}

    if cfg.use_image:
        input_shapes["observation.images.wrist_cam"] = [3, 224, 224]
        input_norms["observation.images.wrist_cam"]  = "mean_std"

    kwargs = dict(
        chunk_size=cfg.chunk_size,
        n_obs_steps=1,
        input_shapes=input_shapes,
        output_shapes={"action": [cfg.action_dim]},
        input_normalization_modes=input_norms,
        output_normalization_modes={"action": "mean_std"},
        dim_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        n_encoder_layers=cfg.num_encoder_layers,
        n_decoder_layers=cfg.num_decoder_layers,
        n_vae_encoder_layers=cfg.num_encoder_layers,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        kl_weight=cfg.kl_weight,
        use_vae=True,
    )

    if cfg.use_image:
        kwargs["vision_backbone"] = "resnet18"
        kwargs["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"

    return _LRConfig(**kwargs)


def _dataset_stats(cfg: ACTConfig, stats: dict) -> dict:
    ds = {
        "observation.state": {
            "mean": torch.from_numpy(stats["state_mean"]),
            "std":  torch.from_numpy(stats["state_std"]),
        },
        "action": {
            "mean": torch.from_numpy(stats["action_mean"]),
            "std":  torch.from_numpy(stats["action_std"]),
        },
    }
    if cfg.use_image:
        # dataset.py already applies ImageNet normalisation — tell lerobot to leave images alone
        ds["observation.images.wrist_cam"] = {
            "mean": torch.zeros(3, 1, 1),
            "std":  torch.ones(3, 1, 1),
        }
    return ds


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig, stats: dict):
        super().__init__()
        self.cfg    = cfg
        self.policy = ACTPolicy(
            _lerobot_config(cfg),
            dataset_stats=_dataset_stats(cfg, stats),
        )

    def _make_batch(self, obs_state, actions=None, action_is_pad=None, obs_image=None):
        batch = {"observation.state": obs_state.unsqueeze(1)}  # (B, 1, state_dim)
        if obs_image is not None:
            batch["observation.images.wrist_cam"] = obs_image.unsqueeze(1)  # (B, 1, C, H, W)
        if actions is not None:
            batch["action"]        = actions
            batch["action_is_pad"] = action_is_pad
        return batch

    def forward(self, obs_state, actions, action_is_pad, obs_image=None):
        loss_dict = self.policy.forward(
            self._make_batch(obs_state, actions, action_is_pad, obs_image)
        )
        l1 = loss_dict.get("l1_loss", loss_dict["loss"])
        kl = loss_dict.get("kl_loss", torch.zeros(1, device=obs_state.device))
        return l1, kl

    def reset(self):
        """Call once at the start of each episode before running predict()."""
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None):
        return self.policy.select_action(
            self._make_batch(obs_state, obs_image=obs_image)
        )
