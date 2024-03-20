import os
import shutil


source_dir = "/media/wenuka/New Volume-G/01.FYP/Dataset/512"
dest_dir = "/media/wenuka/New Volume-G/01.FYP/Dataset/512_splitted/val"

file_path = "/media/wenuka/New Volume-G/01.FYP/Tool_1/Split_dataset/val.txt"

file = open(file_path, "r")
to_embed = [line.strip() for line in file.readlines()]
file.close()

to_move = []
all_names = os.listdir(source_dir)
print("Total: ", len(all_names))

for name in all_names:
    if not (
        "0111" in name.split("&")[0]
        or "0091" in name.split("&")[0]
        or "0096" in name.split("&")[0]
    ):
        if (name.split(".")[0] + "_patch.npy") in to_embed:
            to_move.append(name)
    else:
        tmp_name = name[:-4]
        if tmp_name in to_embed:
            to_move.append(name)

count = 0
for file in to_embed:
    source = os.path.join(source_dir, file)
    dest = os.path.join(dest_dir, file)

    # If dest doesn't exist
    if os.path.exists(dest):
        print("Already exists: ", file)
        continue
    shutil.copy(source, dest)
    count += 1
    print("Moved: ", file)

print("Total moved: ", count)
