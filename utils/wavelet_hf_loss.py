import torch
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward

class HF_consistency(torch.nn.Module):
    def __init__(self):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dwt = DWTForward(J=1, mode='reflect', wave='haar').to(device)


    def forward(self, pred, gt):
        # pred/gt shape:
        B, T, C, H, W = pred.shape
        
        # 2. Flatten Batch and Time to process frames spatially
        # New shape:
        pred_flat = pred.view(-1, C, H, W).float()
        gt_flat = gt.view(-1, C, H, W).float()

        # 3. Apply DWT to extract coefficients
        # coeffs = Low Frequency (LL)
        # coeffs = List of High Frequency [LH, HL, HH]
        pred_coeffs = self.dwt(pred_flat)
        gt_coeffs = self.dwt(gt_flat)

        # 4. Extract and concatenate the High Frequency bands
        # Each band in coeffs is
        pred_hf = torch.cat(pred_coeffs[1], dim=1) 
        gt_hf = torch.cat(gt_coeffs[1], dim=1)     
        hf_mse = F.mse_loss(pred_hf, gt_hf)

        # 5. Calculate MSE on these high-frequency details
        return hf_mse