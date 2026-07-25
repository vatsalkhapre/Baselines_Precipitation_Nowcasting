"""Ablation 2 - Wavelet + MLP (separate MLP per wavelet subband). Loss: MSE."""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=2, loss_type='mse', **kwargs)
