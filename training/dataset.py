import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VIDEO_KEY = "observation.images.wrist_cam"


class FR5Dataset(Dataset):
    def __init__(self, root, chunk_size=100, use_image=True,
                 image_size=(224, 224), episode_indices=None):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.use_image = use_image
        self.image_size = image_size

        with open(self.root / "meta/info.json") as f:
            self.info = json.load(f)

        self.df = pq.read_table(
            self.root / "data/chunk-000/file-000.parquet"
        ).to_pandas()

        self.episodes = pq.read_table(
            self.root / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pandas()

        if episode_indices is not None:
            self.episodes = self.episodes[
                self.episodes["episode_index"].isin(episode_indices)
            ].reset_index(drop=True)

        self._samples = self._build_index()

        self._img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _build_index(self):
        samples = []
        for _, ep in self.episodes.iterrows():
            ep_idx  = int(ep["episode_index"])
            from_i  = int(ep["dataset_from_index"])
            to_i    = int(ep["dataset_to_index"])
            for t in range(from_i, to_i):
                samples.append((ep_idx, t, from_i, to_i))
        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        ep_idx, frame_abs, ep_from, ep_to = self._samples[idx]

        row = self.df.iloc[frame_abs]
        state = torch.tensor(row["observation.state"], dtype=torch.float32)

        chunk_end = min(frame_abs + self.chunk_size, ep_to)
        chunk_rows = self.df.iloc[frame_abs:chunk_end]
        actions = torch.tensor(
            np.array(chunk_rows["action"].tolist()), dtype=torch.float32
        )

        pad_len = self.chunk_size - len(actions)
        is_pad  = torch.zeros(self.chunk_size, dtype=torch.bool)
        if pad_len > 0:
            padding = actions[-1:].expand(pad_len, -1)
            actions = torch.cat([actions, padding], dim=0)
            is_pad[self.chunk_size - pad_len:] = True

        sample = {
            "observation.state": state,
            "action":            actions,
            "action_is_pad":     is_pad,
        }

        if self.use_image:
            frame_in_ep = int(row["frame_index"])
            img = self._load_frame(ep_idx, frame_in_ep)
            if img is not None:
                sample["observation.image"] = img

        return sample

    def _load_frame(self, ep_idx, frame_idx):
        vid = self.root / "videos" / VIDEO_KEY / "chunk-000" / f"file-{ep_idx:03d}.mp4"
        if not vid.exists():
            return None

        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return self._img_transform(img)

    def get_stats(self):
        ep_mask = self.df["episode_index"].isin(
            self.episodes["episode_index"].tolist()
        )
        sub = self.df[ep_mask]
        state  = np.array(sub["observation.state"].tolist())
        action = np.array(sub["action"].tolist())
        return {
            "state_mean":  state.mean(0).astype(np.float32),
            "state_std":   state.std(0).clip(1e-6).astype(np.float32),
            "action_mean": action.mean(0).astype(np.float32),
            "action_std":  action.std(0).clip(1e-6).astype(np.float32),
        }

    @staticmethod
    def episode_split(n_episodes, val_frac=0.1, seed=42):
        indices = list(range(n_episodes))
        random.seed(seed)
        random.shuffle(indices)
        n_val = max(1, int(val_frac * n_episodes))
        return indices[n_val:], indices[:n_val]
