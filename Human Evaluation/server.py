import os
import random
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')
real_images = []
fake_images = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global real_images
    global fake_images
    
    if request.method == 'POST':

        real_images_1 = request.form.getlist('real_images')
        fake_images_1 = request.form.getlist('fake_images')
        
        real_images.extend(real_images_1)
        fake_images.extend(fake_images_1)
        
        print("Real_images : ", real_images)
        print("Fake images : ", fake_images)
        
    image_dir = 'static'

    subfolders = ['cycle_gan', 'diffusion', 'pix2pix' , 'truth']

    image_files = os.listdir(os.path.join(image_dir, subfolders[0]))
    image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    num_images = 40  # Change this to the number of images you want to display
    images =  random.sample(image_files, num_images)

    n_images = num_images / 4
    for i in range(len(images)):
        folder_index = int(i // n_images)
        images[i] = image_dir + "/" + subfolders[folder_index] + "/" + images[i]
    
    random.shuffle(images)
    
    # print("Fake images : ", images)
    print("Total Images: ", len(images))

    real_images = []
    fake_images = []
    return render_template('index.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
