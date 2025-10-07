import os
from glob import glob
import xarray as xr
import numpy as np
import pyart
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
import pickle
import pandas as pd 
from collections import defaultdict
import torch    
from functools import lru_cache
import torchvision.transforms as T
from PIL import Image
from matplotlib import colors
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from utils.dataselection_code import *

PIXEL_SCALE = 60.0
THRESHOLDS = [8, 16, 24, 32]

vil_colors = [
    "#000000",  # black (0)
    "#2020ff",  # blue
    "#00b0f0",  # cyan
    "#00ff80",  # green
    "#a0f000",  # yellow-green
    "#f0d000",  # yellow
    "#f08000",  # orange
    "#f00000",  # red
    "#c00000",  # dark red
    "#a000a0",  # purple
    "#f0f0f0"   # white
]


BOUNDS = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, PIXEL_SCALE]
# =======================All basics========================

def extract_date(filename):
    """Extracts the date part (DDMMMYYYY) from the filename."""
    match = re.match(r"(\d{2}[A-Z]{3}\d{4})_(\d{6})\.nc", filename)
    return match.group(1) if match else None

def sort_files(folder_path):
    """Sorts files in chronological order based on their filenames."""
    files = [f for f in os.listdir(folder_path)]
    
    def extract_sort_key(filename):
        date_part, time_part = filename.split('_')
        months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                  'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
        day, month, year = date_part[:2], months[date_part[2:5]], date_part[5:]
        return f"{year}{month}{day}{time_part[:-3]}"  # YYYYMMDDHHMMSS

    return sorted(files, key=extract_sort_key)

def convert_date(date_str):
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted = date_obj.strftime('%d%b%Y').upper()
    return formatted

def glob_min_max(files, folder_path):
    # Give all files

    all_data = []
    data_path = "/home/vatsal/NWM/weather/Dataset/Full_dataset_240/"
    for file in files:
        file_address = os.path.join(data_path, file)
        print(file_address)
        np_reflec = np.load(file_address, allow_pickle = True)
        np_reflec[np.isnan(np_reflec)] = 0

        all_data.append(np_reflec)
    stacked_all_data = np.stack(all_data, axis=0)
    min = stacked_all_data.min()
    max = stacked_all_data.max()
    with open(os.path.join(folder_path, "min_value.txt"), "w") as file:
        file.write(str(min)) 
    with open(os.path.join(folder_path, "max_value.txt"), "w") as file:
        file.write(str(max)) 
    print("Min max saved")

def save_rainydaysdata_file(mode, data_dir, method = None, csv_file_add = None):
    if mode == "files":
        if method == 0:
            grouped_files = rainy_days(mode, data_dir)
            print(grouped_files)
            
            with open('rainy_days.pkl', 'wb') as f:
                pickle.dump(grouped_files, f)
            print("Rainydays file saved")

        elif method == 1:
            grouped_files = rainy_days_plus_prev_days(mode, data_dir)
            with open('/home/vatsal/MOSDAC/rainy_days_plusprev_days.pkl', 'wb') as f:
                pickle.dump(grouped_files, f)
            print("Rainydays with previous days file saved")
    
    elif mode == 'csv':
        grouped_files = rainy_days(mode, data_dir, csv_file_add)
        print(grouped_files)
        
        with open('rainy_days_IMD.pkl', 'wb') as f:
            pickle.dump(grouped_files, f)
        print("Rainydays file saved")

def compute_min_max(train_files, data_dir):
    print("Min max")
    global_min, global_max = float('inf'), -float('inf')
    for date, files in train_files.items():
        for f in files:
            arr = np.load(os.path.join(data_dir, f + ".npy"))
            local_min, local_max = arr.min(), arr.max()
            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max
    return global_min, global_max

    
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop training if patience exceeded
    
def gray2color(image, **kwargs):

 
    cmap = ListedColormap(vil_colors, name='vil_colormap')
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)
   
    # colored_image = cmap(norm(image))

    
    return image
# ==============================Preprocessing part========================================

def preprocessing_radar(ds, min = None, max = None):
    
    np_reflec = np.array(ds)
    if min is not None and max is not None:
        # assert np_reflec.shape == min.shape, "Shape mismatch between data and mean"
        np_reflec = (np_reflec - min) / (max - min)
        np_reflec[np.isnan(np_reflec)] = 0
        np_reflec = np.clip(np_reflec, 0, 1)  
    else:
        raise ValueError("Data not normalized")
    return np_reflec



# ==================================Dataset class===========================================

class RTimeSeriesDataset(Dataset):

    def __init__(self, file_pairs,data_dir,img_size, min=None, max=None):
        self.min = min
        self.max = max
        self.data_dir = data_dir
        self.file_pairs = file_pairs  # List of (input_files, target_files)
        # Precompute all file paths
        self.all_files = set() # Set avoids duplicates, useful in tracking unique files
        for batch_files in file_pairs:
            self.all_files.update(batch_files)
        self.img_size = img_size
        # Preload metadata or cache frequently used files
        self.cache = {} 

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),  # or BICUBIC
            T.ToTensor()
        ])

    def __len__(self):
        # Total samples minus the sequence lengths needed for input and output
        # return self.num_sequences
        return len(self.file_pairs)
        
    def __getitem__(self, idx):
        
        batch_files = self.file_pairs[idx]
        print(batch_files[0])
        # Load input sequence
        data_point = [self._load_file(f) for f in batch_files]
        
        return torch.stack(data_point)
    
 
    def _load_file(self, filename):
        """Cached file loading with lazy preprocessing"""
        # if filename not in self.cache:
        path = os.path.join(self.data_dir, filename)
        data_array = np.load(path, allow_pickle = True)
        data_array = preprocessing_radar(data_array, self.min, self.max)
        if data_array.shape[0] != self.img_size or data_array.shape[1] != self.img_size:
            
            data_array = torch.tensor(data_array, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
            
            data_array = F.interpolate(data_array, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
            # self.cache[filename] = torch.tensor(data_array, dtype = torch.float32)
            # return self.cache[filename]      
             
            return data_array
        # self.cache[filename] = torch.tensor(data_array, dtype = torch.float32).unsqueeze(0)
        # return self.cache[filename]
        data_array = torch.tensor(data_array, dtype = torch.float32)
        
        return data_array.unsqueeze(0)
    
# ============================== Class to return dataset sequences ========================================

def rainy_dataset(data_dir, input_seq_length, output_seq_length,file_rain_seq_add, img_size, preprocessing):

    # print("inside rainy dataset class")

    grouped_files = {}

    train_dic = {}
    val_dic = {}
    test_dic = {}
    with open(file_rain_seq_add, 'rb') as f:
        grouped_files = pickle.load(f)

    
    # grouped_files = dict(list(grouped_files.items()))

    # Ensure deterministic ordered keys (if pickle stored OrderedDict this keeps order)
    # If it's a plain dict but keys were created in sorted order, preserve order by sorting keys:
    if not isinstance(grouped_files, (dict, OrderedDict)):
        grouped_files = dict(grouped_files)

    keys_sorted = list(grouped_files.keys())
    
    # Seperating the train, test, validation files for the entire dataset
    
    train_files = {}
    validation_items, test_items = [], []

    for date in keys_sorted:
        dt = datetime.strptime(date, "%d%b%Y")
        year = dt.year

        # Assign based on year
        if year == 2018:
            test_items.append((date, grouped_files[date]))
        elif year == 2024:
            validation_items.append((date, grouped_files[date]))
        else:
            train_files[date] = grouped_files[date]
       
    def generate_sequences_list(items, in_len, out_len):
        seqs = []
        for date, files in items:
            n = len(files)
            total = in_len + out_len
            if n < total:
                continue
            for i in range(0, n - total + 1):
                seqs.append(files[i:i + total])
        return seqs

    # Uncomment when train min max needed
    mini, maxi = compute_min_max(train_files, data_dir)
    with open('Min_max_value/reflctivity_min_max.pkl', 'wb') as f:
        pickle.dump({'min': mini, 'max': maxi}, f)


    if preprocessing == 0:
        with open('Min_max_value/reflctivity_min_max.pkl', 'rb') as f:
            stats = pickle.load(f)
        mini, maxi = stats['min'], stats['max']

    train_seq = generate_sequences_list(list(train_files.items()), input_seq_length, output_seq_length)
    val_seq = generate_sequences_list(validation_items, input_seq_length, output_seq_length)
    test_seq = generate_sequences_list(test_items, input_seq_length, output_seq_length)

    train_dataset = RTimeSeriesDataset(train_seq, data_dir, img_size, mini, maxi)
    val_dataset = RTimeSeriesDataset(val_seq, data_dir, img_size, mini, maxi)
    test_dataset = RTimeSeriesDataset(test_seq, data_dir, img_size, mini, maxi)

    return train_dataset, val_dataset, test_dataset

# ==================================Dataloader class===========================================

def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,        # Parallel loading
        pin_memory=True,      # Faster GPU transfers
        persistent_workers=True  # Maintain worker pool
    )


# ========================================End=====================================================