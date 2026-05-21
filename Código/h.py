import numpy as np
sig = np.load("Processed_Data/signals.npy")
met = np.load("Processed_Data/metadata.npy", allow_pickle=True)
suffix = "1"
print("Shape signals:", np.shape(sig))
