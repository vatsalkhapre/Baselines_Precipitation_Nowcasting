import torch
import h5py


data_path = "/home/vatsal/NWM/Dataset/Shanghai_Radar/shanghai.h5"
with h5py.File(data_path,'r') as f:
    imgs = f["train"]["998"] # numpy array: (25, 565, 784), dtype=uint8, range(0,70)
    
    frames = torch.from_numpy(imgs).float()
    print(frames.size())