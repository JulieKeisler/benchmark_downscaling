"""Per-dataset model/training hyperparameters."""

_MODEL_CFG = {
    "MNIST": dict(base_ch=64, ch_mult=(1, 2, 2)),
    "ERA5": dict(base_ch=64, ch_mult=(1, 2, 3), num_res_blocks=3, attention_type="local_cross", time_factor=2),
    "SAFRAN": dict(base_ch=32, ch_mult=(1, 2), attention_resolutions=(), num_res_blocks=2, dropout=0.0),
    "CIFAR10": dict(base_ch=128, ch_mult=(1, 2, 2, 4, 4), num_res_blocks=3, attention_resolutions=(14, 28, 56)),
}

_TRAIN_CFG = {
    "MNIST": dict(batch_size=256, epochs=200, lr=2e-4, accum_iter=1),
    "ERA5": dict(batch_size=64, batch_size_inf=256, epochs=400, lr=2e-4, accum_iter=1),
    "CIFAR10": dict(batch_size=32, epochs=100, lr=2e-4, accum_iter=4),
    "SAFRAN": dict(batch_size=128, batch_size_inf=1024, epochs=200, lr=2e-4, accum_iter=1),
}


def make_model_cfg(data_type: str) -> dict:
    if data_type not in _MODEL_CFG:
        raise ValueError(f"Unknown dataset type: {data_type}")
    return dict(_MODEL_CFG[data_type])


def make_train_cfg(data_type: str) -> dict:
    if data_type not in _TRAIN_CFG:
        raise ValueError(f"Unknown dataset type: {data_type}")
    return dict(_TRAIN_CFG[data_type])
