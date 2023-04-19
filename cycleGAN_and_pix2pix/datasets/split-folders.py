import os
import shutil

folder_path = "./color_5000"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

total = len(images)

train_split, test_split, val_split = 0.5, 0.3, 0.2

train_end = total*train_split
test_end = total*(train_split + test_split)

new_path = os.path.join(folder_path, 'train')
if not os.path.exists(new_path):
    os.makedirs(new_path)

new_path = os.path.join(folder_path, 'test')
if not os.path.exists(new_path):
    os.makedirs(new_path)

new_path = os.path.join(folder_path, 'val')
if not os.path.exists(new_path):
    os.makedirs(new_path)

idx = 0

for image in images:
    old_image_path = os.path.join(folder_path, image)

    if idx < train_end:
        new_image_path = os.path.join(folder_path + "/train", image)
    elif idx < test_end:
        new_image_path = os.path.join(folder_path + "/test", image)
    else:
        new_image_path = os.path.join(folder_path + "/val", image) 
    
    shutil.move(old_image_path, new_image_path)

    idx += 1