"""Ablation 3 - Wavelet + Gabor + MLP, no SRST (srst_depth=0). Loss: MSE.

Equivalent to alpha_amplinet_latent_FAL_FCL_..._expgabor_nosrst_mse_final.py.
"""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=3, loss_type='mse', **kwargs)
