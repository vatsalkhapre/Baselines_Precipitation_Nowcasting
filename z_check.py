import h5py

# Replace 'your_file.h5' with the path to your .h5 file
file_path = '/home/vatsal/Dataserver/Datasets/sevir_lr_latent_32_normalize_resize/data/vil_latent/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5'

# Open the .h5 file in read mode
with h5py.File(file_path, 'r') as h5_file:
    # List all groups in the file
    print("Keys in the file:", list(h5_file.keys()))
    print(h5_file['vil_latent'])
    # Access a specific dataset (replace 'dataset_name' with the actual name)
    # dataset = h5_file['dataset_name']
    # print(dataset[:])  # Print the dataset content