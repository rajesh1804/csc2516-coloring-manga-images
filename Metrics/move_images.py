import os

import shutil

src_folder_path = "./test_latest/images"

dst_folder_path = "./test_latest/images_final"

folder_names = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/Metrics/diff_test"

images = [f for f in os.listdir(folder_names) if os.path.isfile(os.path.join(folder_names, f))]

new_path = os.path.join(dst_folder_path, 'real_A')
if not os.path.exists(new_path):
    os.makedirs(new_path)

new_path = os.path.join(dst_folder_path, 'real_B')
if not os.path.exists(new_path):
    os.makedirs(new_path)

new_path = os.path.join(dst_folder_path, 'fake_B')
if not os.path.exists(new_path):
    os.makedirs(new_path)

subfolders = ['real_A', 'real_B', 'fake_B']


for image in images:
    for folder in subfolders:
        old_image_path = os.path.join(src_folder_path, folder, image)
        new_image_path = os.path.join(dst_folder_path, folder, image)

        # print(old_image_path)

        # img_cls = '_fake_B.png'
        # if old_image_path.endswith(img_cls):
        #     new_image_path = os.path.join(folder_path + "/fake_B", image.removesuffix(img_cls)+ ".png")

        # img_cls = '_real_B.png'
        # if old_image_path.endswith(img_cls):
        #     new_image_path = os.path.join(folder_path + "/real_B", image.removesuffix(img_cls)+ ".png")

        # img_cls = '_real_A.png'
        # if old_image_path.endswith(img_cls):
        #     new_image_path = os.path.join(folder_path + "/real_A", image.removesuffix(img_cls) + ".png")
        
        shutil.copy(old_image_path, new_image_path)