"""Ablation 6 - Full model (srst_depth=2) trained with FACL loss (RandomScheduling)."""
from models.ablations_neurips_rebuttal._dawncast_common import make_model


def get_model(**kwargs):
    return make_model(ablation_id=6, loss_type='facl', **kwargs)
