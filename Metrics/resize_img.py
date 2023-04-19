from PIL import Image
import os

diff_folder = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/Metrics/diff_test"
diff_folder_new = "/Users/naveenthangavelu/Documents/MScAC/Neural networks and Deep Learning/Project/Metrics/diff_test_resized"

if not os.path.exists(diff_folder_new):
    os.makedirs(diff_folder_new)

images = [f for f in os.listdir(diff_folder) if os.path.isfile(os.path.join(diff_folder, f))]

for img in images:
    # Open the image file
    old_image_path = os.path.join(diff_folder, img)
    image = Image.open(old_image_path)

    # Resize the image
    resized_image = image.resize((256, 256))

    # Save the resized image
    resized_image.save(os.path.join(diff_folder_new, img))