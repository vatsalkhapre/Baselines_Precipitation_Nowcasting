import os
from glob import glob
import xarray as xr
import numpy as np
import pyart
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict,  OrderedDict
import re
import pickle
import pandas as pd 
from collections import defaultdict
import torch    
from functools import lru_cache
import torchvision.transforms as T
from PIL import Image
from matplotlib import colors
from matplotlib.colors import ListedColormap , BoundaryNorm
import torch.nn.functional as F
from copy import deepcopy
from utils.dataselection_code import *

PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
THRESHOLDS = (16, 74, 133, 160, 181, 219)
VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
VIL_COLORS = [
            [0, 0, 0],
            [0.30196078, 0.30196078, 0.30196078],
            [0.15686275, 0.74509804, 0.15686275],
            [0.09803921, 0.58823529, 0.09803921],
            [0.03921568, 0.41176471, 0.03921568],
            [0.03921568, 0.29411765, 0.03921568],
            [0.96078431, 0.96078431, 0.0],
            [0.92941176, 0.67450980, 0.0],
            [0.94117647, 0.43137255, 0.0],
            [0.62745098, 0.0, 0.0],
            [0.90588235, 0.0, 1.0]
        ]

cols = deepcopy(VIL_COLORS)
lev = deepcopy(VIL_LEVELS)
nil, under, over = cols.pop(0), cols[0], cols[-1]
cmap = ListedColormap(cols)
cmap.set_bad(nil); cmap.set_under(under); cmap.set_over(over)
norm = BoundaryNorm(lev, cmap.N)

# =======================All basics========================

def sort_files(folder_path):
    """Sorts files in chronological order based on their filenames."""
    files = [f for f in os.listdir(folder_path)]
    
    def extract_sort_key(filename):
        date_part, time_part = filename.split('_')
        months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                  'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
        day, month, year = date_part[:2], months[date_part[2:5]], date_part[5:]
        return f"{year}{month}{day}{time_part[:-3]}"  # YYYYMMDDHHMMSS

def convert_date(date_str):
    
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted = date_obj.strftime('%d%b%Y').upper()
    return formatted



def gray2color(image, **kwargs):

 
    VIL_COLORS = [
            [0, 0, 0],
            [0.30196078, 0.30196078, 0.30196078],
            [0.15686275, 0.74509804, 0.15686275],
            [0.09803921, 0.58823529, 0.09803921],
            [0.03921568, 0.41176471, 0.03921568],
            [0.03921568, 0.29411765, 0.03921568],
            [0.96078431, 0.96078431, 0.0],
            [0.92941176, 0.67450980, 0.0],
            [0.94117647, 0.43137255, 0.0],
            [0.62745098, 0.0, 0.0],
            [0.90588235, 0.0, 1.0]
        ]
    VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]


    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)

    nil = cols.pop(0)
    under = cols[0]
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
   
    colored_image = cmap(norm(image))

    
    return colored_image

def preprocessing_vil(vil, mini = None, maxi = None):
    
    if mini is not None and maxi is not None:
        # assert np_reflec.shape == min.shape, "Shape mismatch between data and mean"
        vil_scaled = (vil - mini) / (maxi- mini)
    
        return np.clip(vil_scaled, 0, 1)

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

# ==================================Dataset class===========================================

class VTimeSeriesDataset(Dataset):

    def __init__(self, file_pairs,data_dir,img_size, min_val=None, max_val=None, max_cache_size=1024, use_mmap=True):
        self.min = min_val
        self.max = max_val
        self.data_dir = data_dir
        self.file_pairs = file_pairs  # List of (input_files, target_files)
        self.img_size = img_size
        self.use_mmap = use_mmap
        
        # small simple LRU-ish cache (evict oldest inserted when over capacity)
        self.cache = {} 
        self.cache_access_order = []  # list of keys for eviction
        self.max_cache_size = max_cache_size
        

    def __len__(self):
        # Total samples minus the sequence lengths needed for input and output
        # return self.num_sequences
        return len(self.file_pairs)


    def __getitem__(self, idx):
        batch_files = self.file_pairs[idx]
        data_point = [self._load_file(f) for f in batch_files]
         # each => torch.Tensor (1,H,W)
        # Stack into (T, C, H, W) or (T, 1, H, W) depending on your model (here T,1,H,W)
        return torch.stack(data_point)

        
    def _maybe_cache_put(self, key, value):
        if self.max_cache_size <= 0:
            return
        if key in self.cache:
            # update access order
            try:
                self.cache_access_order.remove(key)
            except ValueError:
                pass
            self.cache_access_order.append(key)
            self.cache[key] = value
            return
        # insert
        self.cache[key] = value
        self.cache_access_order.append(key)
        if len(self.cache_access_order) > self.max_cache_size:
            # evict oldest
            oldest = self.cache_access_order.pop(0)
            del self.cache[oldest]


    def _load_file(self, filename):
 
        if filename.lower().endswith('.npy'):
            key = filename  # key for cache
            path = os.path.join(self.data_dir, filename)
        else:
            key = filename + '.npy'
            path = os.path.join(self.data_dir, key)
        
        if key in self.cache:
            return self.cache[key]

        try:
            arr = np.load(path)
            arr = arr.astype(np.float32, copy=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")

        # Apply preprocessing function (assumed to be numpy -> numpy)
        arr = preprocessing_vil(arr, self.min, self.max)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        # convert to torch without extra copy: torch.from_numpy (arr must be contiguous)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        tensor = torch.from_numpy(arr)  # shape (H,W) usually
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1,H,W)
        tensor = tensor.float()

        # Resize if necessary to (1, img_size, img_size)
        _, h, w = tensor.shape
        if (h != self.img_size) or (w != self.img_size):
            # add batch dim for interpolate
            t = tensor.unsqueeze(0)  # (1, 1, H, W)
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            tensor = t.squeeze(0) # (1, Hnew, Wnew)
    
        # store in cache
        self._maybe_cache_put(key, tensor)
        
        return tensor

# ============================== Class to return dataset sequences ========================================

def rainy_dataset(data_dir, input_seq_length, output_seq_length, file_rain_seq_add, img_size, preprocessing):
    
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

    # # Uncomment when train min max needed
    # mini, maxi = compute_min_max(train_files, data_dir)
    # with open('vil_vip_min_max.pkl', 'wb') as f:
    #     pickle.dump({'min': mini, 'max': maxi}, f)
    # exit()

    if preprocessing == 0:
        with open('Min_max_value/vil_vip_min_max.pkl', 'rb') as f:
            stats = pickle.load(f)
        mini, maxi = stats['min'], stats['max']

    train_seq = generate_sequences_list(list(train_files.items()), input_seq_length, output_seq_length)
    val_seq = generate_sequences_list(validation_items, input_seq_length, output_seq_length)
    test_seq = generate_sequences_list(test_items, input_seq_length, output_seq_length)

    train_dataset = VTimeSeriesDataset(train_seq, data_dir, img_size, mini, maxi)
    val_dataset = VTimeSeriesDataset(val_seq, data_dir, img_size, mini, maxi)
    test_dataset = VTimeSeriesDataset(test_seq, data_dir, img_size, mini, maxi)

    return train_dataset, val_dataset, test_dataset

# ======================================Dataloader class=============================================

def create_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,        # Parallel loading
        pin_memory=True,      # Faster GPU transfers
        persistent_workers=True  # Maintain worker pool
    )


# ============================================End=====================================================