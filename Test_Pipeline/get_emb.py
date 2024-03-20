import os
import numpy as np
from preprocess import preprocess_image
import tensorflow as tf
import matplotlib.pyplot as plt

saved_model_path = (
    "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"
)
saved_model = tf.saved_model.load(saved_model_path)

patch_path = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/patches/0"
all_names = os.listdir(patch_path)

for file in all_names:
    print(f"processing {file}")
    patches = np.load(os.path.join(patch_path, file))
    per_img_embed = []
    for patch_3d in range(patches.shape[0]):
        per_patch_emb = []
        selected_patch = patches[patch_3d]
        preprocessed_patch = preprocess_image(
            selected_patch, 32, 32, is_training=False, color_distort=False
        )
        preprocessed_patch_batched = np.expand_dims(preprocessed_patch, axis=0)
        embeddings = saved_model(preprocessed_patch_batched, trainable=False)[
            "final_avg_pool"
        ]
        per_img_embed.append(embeddings)
        output_file_path = os.path.join(
        "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/resnet_emb/0",
        file,
    )
   
    np.save(output_file_path, per_img_embed)
    print(f"Embeds done {file}")
 
 