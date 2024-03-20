import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


loaded_numpy_data_dict = np.load(
    "attention_weights_resnet_4096_dimrNone_train.npy", allow_pickle=True
).item()

keys = loaded_numpy_data_dict.keys()

keys = list(keys)
# print(keys)
# print(len(keys))

# # name = keys[10]
# # print(name)


for key in keys:
    name = key

    # print(name)
    vals = loaded_numpy_data_dict[name]
    # print(len(vals))

    tier2_weights = vals["tier 2"]
    # print(tier2_weights.shape)

    tensor = np.zeros(512)
    for i in range(512):
        bag_idx = vals[i][1]
        tensor[i] = vals[i][0]
        # *tier2_weights[0][bag_idx]
        tensor[i] = tensor[i] * 1000  # Multiple by 10^3

    # print(tensor.shape)
    # print(tensor)

    # Reshaping the NumPy array
    reshaped_attention = tensor.reshape(8, 8, 8)

    # Average across the Z-dimension (third dimension)
    # average_attention = np.mean(reshaped_attention, axis=2)
    # average_attention = average_attention.T

    # print(average_attention.shape)

    for i in range(reshaped_attention.shape[2]):
        slice = reshaped_attention[:, :, i]
        slice = slice.T

        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            slice,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Attention"},
        )
        ax.invert_yaxis()
        plt.title("Attention Across Patches")
        plt.savefig(
            f"Attention_maps_train/1st Tier/All/{name}_slice{i}.png"
        )
    # plt.savefig(f"Attention_maps_train/Both Tiers/AVG/{name}.png")
    # plt.close()
        plt.close()
    # plt.show()