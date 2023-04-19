import os
import shutil

# Set the source directory where the images are located
src_dir = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/best"

# Set the destination directory where the renamed images will be copied
dst_dir = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/Metrics/diff_test_new"


# Create the destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    # Check if the file is an image with the name format Teppei_Arima_*.png
    if filename.endswith(".jpg"):
        # Extract the name without the number and extension
        name = filename.split("_")[:-1]
        name = "_".join(name) + ".jpg"
        # Copy the file to the destination directory with the new name
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, name))