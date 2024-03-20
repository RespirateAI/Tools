import os
import shutil
from sklearn.model_selection import train_test_split

# Define directories for the two classes
class1_dir = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/resnet_emb/0"
class2_dir = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/resnet_emb/1"

# Define output directories for train, validation, and test splits
output_train_dir = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/splitted/val"
output_val_dir = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/splitted/train"
output_test_dir = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/splitted/test"

# Get the list of filenames for each class
class1_files = [os.path.join(class1_dir, filename) for filename in os.listdir(class1_dir)]
class2_files = [os.path.join(class2_dir, filename) for filename in os.listdir(class2_dir)]

# Perform stratified train-validation-test split for each class
class1_train_val, class1_test = train_test_split(class1_files, test_size=0.2, random_state=42)
class1_train, class1_val = train_test_split(class1_train_val, test_size=0.25, random_state=42)

class2_train_val, class2_test = train_test_split(class2_files, test_size=0.2, random_state=42)
class2_train, class2_val = train_test_split(class2_train_val, test_size=0.25, random_state=42)

# Move files to the appropriate directories
for filename in class1_train:
    shutil.copy(filename, os.path.join(output_train_dir, os.path.basename(filename)))
for filename in class1_val:
    shutil.copy(filename, os.path.join(output_val_dir, os.path.basename(filename)))
for filename in class1_test:
    shutil.copy(filename, os.path.join(output_test_dir, os.path.basename(filename)))

for filename in class2_train:
    shutil.copy(filename, os.path.join(output_train_dir, os.path.basename(filename)))
for filename in class2_val:
    shutil.copy(filename, os.path.join(output_val_dir, os.path.basename(filename)))
for filename in class2_test:
    shutil.copy(filename, os.path.join(output_test_dir, os.path.basename(filename)))

print("Stratified train-validation-test split completed successfully.")
