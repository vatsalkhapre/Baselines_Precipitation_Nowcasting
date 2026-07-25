"""Ablation 4 - Wavelet + Gabor + MLP + 1 SRSTResBlock + STRModule (srst_depth=1). Loss: MSE."""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=4, loss_type='mse', **kwargs)
