import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


loaded_numpy_data_dict = np.load(
    "./Attention/attention_weights_resnet_random_binary_resnet_32_comb_train.npy", allow_pickle=True
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
    tier2_weights = vals["tier 2"]
    print(tier2_weights.shape)

    tensor = np.zeros(64)
    for i in range(64):
        bag_idx = vals[i][1]
        tensor[i] = vals[i][0] * tier2_weights[0][bag_idx]
        tensor[i] = tensor[i] * 1000  # Multiple by 10^3

    # print(tensor.shape)
    # print(tensor)

    # Reshaping the NumPy array
    reshaped_attention = tensor.reshape(8, 8)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        reshaped_attention,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Attention"},
    )
    ax.invert_yaxis()
    plt.title("Attention Across Patches")
    plt.savefig(
        f"Attention_Maps_new/Testing_Pipeline/Resnet/Both Tiers/All/{name}.png"
    )
    plt.close()

