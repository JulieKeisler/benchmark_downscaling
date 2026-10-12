"""Pseudo-pair dataset for conditional flow matching, with DOY + per-sample stats.

Static predictors are NOT included here: they are constant across all
samples and live as a buffer on the model (see model.UNetFlow.set_static).
"""

import torch
from torch.utils.data import Dataset

from .data_utils import normalize_column
from .filters import low_pass_tensor_batch
from .time_utils import build_doy_array


class SerpentFlowDataset(Dataset):
    def __init__(self, input_path, target_path, start_date: str = "1980-01-01",
                 apply_noise: bool = True, dual: bool = False, num_doy: int = 365,
                 noise_scale=(1.0, 1.0, 1.0)):
        super().__init__()
        self.apply_noise = apply_noise
        self.dual = dual
        self.num_doy = num_doy

        self.input_data = torch.load(input_path, map_location="cpu")
        self.target_data = torch.load(target_path, map_location="cpu")
        assert self.input_data.shape == self.target_data.shape

        for tensor in (self.target_data, self.input_data):
            for c in range(tensor.shape[1]):
                tensor[:, c] = normalize_column(tensor[:, c])

        # per-sample low-frequency input statistics, computed on the NORMALIZED
        # data: raw physical units (e.g. psl ~1e5 Pa) overflow fp16 autocast in
        # the stats_proj Linear on GPU (inf -> NaN). train/infer/reconstruction
        # all use this same dataset class, so the change is consistent.
        self.stats_data = torch.stack(
            [self.input_data.mean(dim=(2, 3)), self.input_data.std(dim=(2, 3))], dim=2
        )

        self.noise_scale = torch.tensor(noise_scale, dtype=torch.float32)[:, None, None]
        self.doy = build_doy_array(start_date, len(self.target_data), num_doy)

    def __getitem__(self, index):
        data = self.target_data[index]
        noisy_data = self.input_data[index]
        stats = self.stats_data[index]
        doy = self.doy[index]

        if self.apply_noise:
            _, noise_lr = low_pass_tensor_batch(data, 1.0, apply_noise=True, method="fourier")
            noisy_data = noisy_data + noise_lr * self.noise_scale.to(noise_lr.device)

        if self.dual:
            noisy_data = torch.randn_like(noisy_data)

        return {"noisy": noisy_data, "data": data, "stats": stats, "doy": doy}

    def __len__(self):
        return len(self.target_data)
