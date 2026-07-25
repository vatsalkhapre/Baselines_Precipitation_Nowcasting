"""Ablation 1 - MLP only (no Wavelet, no Gabor, no SRST). Loss: MSE."""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=1, loss_type='mse', **kwargs)
