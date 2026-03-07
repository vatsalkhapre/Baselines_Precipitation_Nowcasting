"""
=============================================================================
RESULTS LOGGER - Auto-save evaluation results to shared CSV
=============================================================================

Uses CSV format which:
- Works on any Linux system without MS Excel
- Can be opened in Excel, Google Sheets, LibreOffice Calc
- Can be viewed with cat, less, column -t, or any text editor

USAGE:
    from utils.results_logger import ResultsLogger
    
    res_dict = eval.done()
    
    logger = ResultsLogger(csv_path="/path/to/shared/results.csv")
    logger.log_results(
        res_dict=res_dict,
        backbone=self.args.backbone,
        exp_note=self.args.exp_note,
        dataset=self.args.dataset,
    )

=============================================================================
"""

import os
import csv
import socket
import fcntl
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional


class ResultsLogger:
    """
    Logger that saves evaluation results to a shared CSV file.
    Supports concurrent access from multiple GPUs/servers using file locking.
    """
    
    COLUMNS = [
        "Date",
        "Time", 
        "GPU",
        "Model",
        "Dataset",
        "Experiment Details",
        "Why?",
        "CSI-M",
        "CSI-4",
        "CSI-16",
        "HSS",
        "SSIM",
        "MSE",
        "PSNR",
        "MAE",
        "RMSE",
        "CRPS",
        "LPIPS",
    ]
    
    def __init__(self, csv_path: str = None):
        """
        Initialize the results logger.
        
        Args:
            csv_path: Path to the shared CSV file. 
                      If None, defaults to ~/shared_results/evaluation_results.csv
        """
        if csv_path is None:
            csv_path = os.path.expanduser("~/shared_results/evaluation_results.csv")
        
        self.csv_path = csv_path
        self._ensure_directory()
        self._ensure_csv_exists()
    
    def _ensure_directory(self):
        """Create directory if it doesn't exist."""
        directory = os.path.dirname(self.csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
            print(f"[ResultsLogger] Created new CSV file: {self.csv_path}")
    
    def _get_real_ip(self) -> str:
        """
        Get the actual network IP address (not localhost).
        Tries multiple methods to find the real IP.
        """
        # Method 1: Connect to external address (doesn't actually send data)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        
        # Method 2: Use hostname command
        try:
            result = subprocess.run(
                ["hostname", "-I"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                for ip in ips:
                    if ip and not ip.startswith("127."):
                        return ip
        except Exception:
            pass
        
        # Method 3: Parse ip addr output
        try:
            result = subprocess.run(
                ["ip", "addr"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'inet ' in line and '127.0.0.1' not in line:
                        # Extract IP from line like "    inet 10.24.52.88/24 ..."
                        parts = line.strip().split()
                        for i, part in enumerate(parts):
                            if part == 'inet' and i + 1 < len(parts):
                                ip = parts[i + 1].split('/')[0]
                                if not ip.startswith("127."):
                                    return ip
        except Exception:
            pass
        
        # Method 4: Fallback to gethostbyname
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if not ip.startswith("127."):
                return ip
        except Exception:
            pass
        
        return "unknown"
    
    def _get_gpu_identifier(self) -> str:
        """
        Get GPU/server identifier.
        Returns the last octet of IP address (e.g., '.88' from '10.24.52.88')
        """
        ip = self._get_real_ip()
        if ip and ip != "unknown":
            last_octet = ip.split('.')[-1]
            return f".{last_octet}"
        return "unknown"
    
    def _get_cuda_device(self) -> str:
        """Get CUDA device ID if available."""
        try:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                return f"cuda:{cuda_visible}"
        except Exception:
            pass
        return ""
    
    def log_results(
        self,
        res_dict: Dict[str, float],
        backbone: str,
        exp_note: str,
        dataset: str = "",
        why: str = "",
        use_full_ip: bool = False,
        include_cuda: bool = True,
    ) -> bool:
        """
        Log evaluation results to the CSV file.
        
        Args:
            res_dict: Dictionary returned by Evaluator.done() containing metrics
            backbone: Model backbone name
            exp_note: Experiment note/details
            dataset: Dataset name
            why: Reason for experiment (optional, fill later)
            use_full_ip: If True, use full IP; otherwise use last octet
            include_cuda: If True, append CUDA device to GPU identifier
        
        Returns:
            True if logging was successful, False otherwise
        """
        now = datetime.now()
        
        # Build GPU identifier
        if use_full_ip:
            gpu_id = self._get_real_ip()
        else:
            gpu_id = self._get_gpu_identifier()
        
        # Optionally append CUDA device
        if include_cuda:
            cuda_dev = self._get_cuda_device()
            if cuda_dev:
                gpu_id = f"{gpu_id} ({cuda_dev})"
        
        row_data = [
            now.strftime("%Y-%m-%d"),                                    # Date
            now.strftime("%H:%M:%S"),                                    # Time
            gpu_id,                                                       # GPU
            backbone,                                                     # Model
            dataset,                                                      # Dataset
            exp_note,                                                     # Experiment Details
            why,                                                          # Why?
            self._format_metric(res_dict, 'csi'),                        # CSI-M
            self._format_metric(res_dict, 'csi4'),                       # CSI-4
            self._format_metric(res_dict, 'csi16'),                      # CSI-16
            self._format_metric(res_dict, 'hss'),                        # HSS
            self._format_metric(res_dict, 'ssim'),                       # SSIM
            self._format_metric(res_dict, 'mse'),                        # MSE
            self._format_metric(res_dict, 'psnr'),                       # PSNR
            self._format_metric(res_dict, 'mae'),                        # MAE
            self._format_metric(res_dict, 'rmse'),                       # RMSE
            self._format_metric(res_dict, 'crps'),                       # CRPS
            self._format_metric(res_dict, 'lpips'),                      # LPIPS
        ]
        
        return self._append_row_with_lock(row_data, backbone)
    
    def _format_metric(self, d: Dict, key: str, precision: int = 6) -> str:
        """Format metric value as string with specified precision."""
        try:
            value = d.get(key)
            if value is None:
                return ""
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return ""
    
    def _append_row_with_lock(self, row_data: list, backbone: str) -> bool:
        """
        Append a row to CSV file with file locking for concurrent access.
        """
        lock_file = self.csv_path + ".lock"
        
        try:
            # Create/open lock file
            with open(lock_file, 'w') as lock_fp:
                # Acquire exclusive lock
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Append row to CSV
                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)
                    
                    print(f"[ResultsLogger] ✅ Results logged to: {self.csv_path}")
                    print(f"[ResultsLogger] Model: {backbone}, CSI-M: {row_data[7]}")
                    return True
                    
                finally:
                    # Release lock
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                    
        except Exception as e:
            print(f"[ResultsLogger] ❌ Failed to log results: {e}")
            self._save_backup(row_data)
            return False
    
    def _save_backup(self, row_data: list):
        """Save to a backup file if main write fails."""
        backup_path = self.csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            with open(backup_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
                writer.writerow(row_data)
            print(f"[ResultsLogger] Backup saved to: {backup_path}")
        except Exception as e:
            print(f"[ResultsLogger] Backup also failed: {e}")
    
    def print_results(self, last_n: int = 10):
        """Print last N results in a formatted table."""
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) <= 1:
                print("No results yet.")
                return
            
            header = rows[0]
            data = rows[1:][-last_n:]
            
            widths = [max(len(str(row[i])) for row in [header] + data) for i in range(len(header))]
            
            print("-" * (sum(widths) + len(widths) * 3))
            print(" | ".join(f"{header[i]:<{widths[i]}}" for i in range(len(header))))
            print("-" * (sum(widths) + len(widths) * 3))
            for row in data:
                print(" | ".join(f"{row[i]:<{widths[i]}}" for i in range(len(row))))
            print("-" * (sum(widths) + len(widths) * 3))
            
        except Exception as e:
            print(f"Error reading results: {e}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def log_evaluation_results(
    res_dict: Dict[str, float],
    backbone: str,
    exp_note: str,
    dataset: str = "",
    csv_path: str = None,
    why: str = "",
):
    """Convenience function to log results."""
    logger = ResultsLogger(csv_path=csv_path)
    return logger.log_results(
        res_dict=res_dict,
        backbone=backbone,
        exp_note=exp_note,
        dataset=dataset,
        why=why,
    )


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    # Test IP detection
    logger = ResultsLogger(csv_path="./test_results.csv")
    
    print(f"Detected IP: {logger._get_real_ip()}")
    print(f"GPU Identifier: {logger._get_gpu_identifier()}")
    print(f"CUDA Device: {logger._get_cuda_device()}")
    
    test_res_dict = {
        'csi': 0.4523,
        'csi4': 0.5234,
        'csi16': 0.6123,
        'hss': 0.3456,
        'ssim': 0.8765,
        'mse': 0.0123,
        'psnr': 25.432,
        'mae': 0.0234,
        'rmse': 0.1109,
        'crps': 0.0456,
        'lpips': 0.1234,
    }
    
    logger.log_results(
        res_dict=test_res_dict,
        backbone="amplinet_latent_falfcl_only_2.3.1",
        exp_note="Test experiment",
        dataset="meteo_lr_latent_32",
    )
    
    print("\n📊 Results:")
    logger.print_results()
