"""Ablation 5 - Full model: Wavelet + Gabor + MLP + 2 SRSTResBlock + STRModule (srst_depth=2). Loss: MSE."""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=5, loss_type='mse', **kwargs)
