import pickle

with open("/home/vatsal/Dataserver/Datasets/VIL/VIL_scaled_lr_240/train_chunks.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data))
