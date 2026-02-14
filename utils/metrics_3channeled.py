import numpy as np
import logging
import lpips
import torch
import cv2
import matplotlib.pyplot as plt

plt.switch_backend('agg')
np.seterr(divide='ignore', invalid='ignore')


def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)


# ===============================
# Max Pool (Channel Safe)
# ===============================
def max_pool(arr, pool_size):
    """
    Supports:
    (H,W)
    (C,H,W)
    """
    if arr.ndim == 2:
        arr = arr[None, ...]

    C, H, W = arr.shape

    pad_H = (pool_size - H % pool_size) % pool_size
    pad_W = (pool_size - W % pool_size) % pool_size

    arr = np.pad(arr, ((0, 0), (0, pad_H), (0, pad_W)))

    H2 = H + pad_H
    W2 = W + pad_W

    arr = arr.reshape(C, H2 // pool_size, pool_size,
                      W2 // pool_size, pool_size)

    arr = arr.transpose(0, 1, 3, 2, 4)

    pooled = np.max(arr, axis=(3, 4))

    return pooled


# ===============================
# SSIM (Channel Safe)
# ===============================
def cal_ssim(pred, true, data_range=255):
    if pred.ndim == 3:
        # Channel-wise SSIM average
        return np.mean([
            cal_ssim(pred[c], true[c], data_range)
            for c in range(pred.shape[0])
        ])

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


# ===============================
# Evaluator
# ===============================
class Evaluator(object):

    def __init__(self, seq_len, value_scale, thresholds):

        self.thresholds = thresholds
        self.seq_len = seq_len
        self.value_scale = value_scale
        self.total = 0

        self.metrics = {}
        for threshold in thresholds:
            self.metrics[threshold] = {
                "hits": [], "misses": [],
                "falsealarms": [], "correctnegs": []
            }

        self.losses = {
            "mse": [], "mae": [], "rmse": [],
            "psnr": [], "ssim": [], "lpips": []
        }

        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            self.lpips_fn.cuda()

    # ===============================
    # Utility
    # ===============================
    def float2int(self, arr):
        x = arr.clip(0.0, 1.0)
        x = x * self.value_scale
        return x.astype(np.uint16)

    def _ensure_channel_dim(self, arr):
        """
        Ensure shape: (B,T,C,H,W)
        """
        if arr.ndim == 4:
            arr = arr[:, :, None, :, :]
        return arr

    # ===============================
    # Threshold Metrics
    # ===============================
    def cal_frame(self, obs, sim, threshold):
        if obs.ndim == 3:
            # average channels for threshold metrics
            obs = obs.mean(axis=0)
            sim = sim.mean(axis=0)

        obs = np.where(obs >= threshold, 1, 0)
        sim = np.where(sim >= threshold, 1, 0)

        hits = np.sum((obs == 1) & (sim == 1))
        misses = np.sum((obs == 1) & (sim == 0))
        falsealarms = np.sum((obs == 0) & (sim == 1))
        correctneg = np.sum((obs == 0) & (sim == 0))

        return hits, misses, falsealarms, correctneg

    # ===============================
    # Frame Losses (Channel Safe)
    # ===============================
    def cal_frame_losses(self, pred, true):

        if pred.ndim == 3:
            metrics = [
                self.cal_frame_losses(pred[c], true[c])
                for c in range(pred.shape[0])
            ]
            return np.mean(metrics, axis=0)

        pred = pred * self.value_scale
        true = true * self.value_scale

        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10(self.value_scale / np.sqrt(mse + 1e-8))
        ssim = cal_ssim(pred, true, self.value_scale)

        return mae, mse, rmse, psnr, ssim

    # ===============================
    # LPIPS (Channel Safe)
    # ===============================
    def cal_batch_lpips(self, preds, trues):

        preds = torch.from_numpy(preds).float()
        trues = torch.from_numpy(trues).float()

        if preds.shape[2] == 1:
            preds = preds.repeat(1, 1, 3, 1, 1)
            trues = trues.repeat(1, 1, 3, 1, 1)

        preds = preds * 2.0 - 1.0
        trues = trues * 2.0 - 1.0

        if torch.cuda.is_available():
            preds = preds.cuda()
            trues = trues.cuda()

        lpips_list = []
        for t in range(preds.shape[1]):
            val = self.lpips_fn(preds[:, t], trues[:, t])
            lpips_list.append(val.detach().cpu().numpy())

        return np.mean(lpips_list, axis=0)

    # ===============================
    # Main Evaluation
    # ===============================
    def evaluate(self, true_batch, pred_batch):

        if isinstance(pred_batch, torch.Tensor):
            pred_batch = pred_batch.detach().cpu().numpy()
            true_batch = true_batch.detach().cpu().numpy()

        pred_batch = self._ensure_channel_dim(pred_batch)
        true_batch = self._ensure_channel_dim(true_batch)

        B, T, C, H, W = true_batch.shape

        pred_batch = pred_batch.clip(0.0, 1.0)
        true_batch = true_batch.clip(0.0, 1.0)

        # LPIPS
        lpips_vals = self.cal_batch_lpips(pred_batch, true_batch)
        self.losses['lpips'].extend(lpips_vals)

        pred_int = self.float2int(pred_batch)
        gt_int = self.float2int(true_batch)

        for b in range(B):
            for t in range(T):

                # threshold metrics
                for threshold in self.thresholds:
                    h, m, f, c = self.cal_frame(
                        gt_int[b, t], pred_int[b, t], threshold
                    )
                    self.metrics[threshold]["hits"].append(h)
                    self.metrics[threshold]["misses"].append(m)
                    self.metrics[threshold]["falsealarms"].append(f)
                    self.metrics[threshold]["correctnegs"].append(c)

                # regression metrics
                mae, mse, rmse, psnr, ssim = self.cal_frame_losses(
                    true_batch[b, t], pred_batch[b, t]
                )

                self.losses['mae'].append(mae)
                self.losses['mse'].append(mse)
                self.losses['rmse'].append(rmse)
                self.losses['psnr'].append(psnr)
                self.losses['ssim'].append(ssim)

        self.total += B

    # ===============================
    # Final Summary
    # ===============================
    def done(self):

        res = {}

        res['mse'] = np.mean(self.losses['mse'])
        res['mae'] = np.mean(self.losses['mae'])
        res['rmse'] = np.mean(self.losses['rmse'])
        res['psnr'] = np.mean(self.losses['psnr'])
        res['ssim'] = np.mean(self.losses['ssim'])
        res['lpips'] = np.mean(self.losses['lpips'])

        print_log("=" * 60)
        for k, v in res.items():
            print_log(f"{k.upper()} : {v}")
        print_log("=" * 60)

        return res
