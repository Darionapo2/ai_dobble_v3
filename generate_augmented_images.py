from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os, shutil

# This script perform a dataset augmentation.
# many slightly distorted or modified variation of each card are generated in order to train the
# model for a more generalistic use case.

deck = 'exp0'
data_path = f'new_dataset/{deck}/'

augmented_imgs_path = f'new_dataset/{deck}-augmented3/'

# checks if the directory exists, if yes, create a new directory
# to store the modified images
if os.path.exists(augmented_imgs_path):
    shutil.rmtree(augmented_imgs_path)
os.mkdir(augmented_imgs_path)

# total variations for each original card of the dataset
total_images_to_augment = 70
# DPI of the final augmented image
image_size = 50

# distortion parameters --> all of them are generated randomly between the selected boundaries
width_shift_range = 0.1
height_shift_range = 0.1
brightness_range = [0.2, 1.1]
zoom_range = [0.7, 1.3]

for folder in os.listdir(data_path):
    print(f'[INFO] generating images in folder {folder}')

    for file in os.listdir(f'{data_path}/{folder}'):

        # Load each image
        img = load_img(f'{data_path}/{folder}/{file}')
        # Convert to numpy array
        data = img_to_array(img)
        # Expand dimension to one sample
        samples = expand_dims(data, 0)

        # Create image data augmentation generator
        datagen = ImageDataGenerator(
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            brightness_range = brightness_range,
            zoom_range = zoom_range
        )

        # Prepare iterator
        it = datagen.flow(samples, batch_size = 1)

        # Path of the specific output directory
        output_path = f'{augmented_imgs_path}{folder}/'

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

        for i in range(1, total_images_to_augment + 1):
            # Generate batch of images
            batch = it.next()
            # Convert to unsigned integers for viewing
            image = batch[0].astype('uint8')

            fig = pyplot.figure(frameon = False)
            ax = pyplot.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # plot raw pixel data
            ax.imshow(image)
            # using the extention .tif we can minimize the storage space used by each image
            saved_filename = f'{output_path}card{folder}' + '_{:03d}.tif'.format(i)
            fig.savefig(fname = saved_filename, dpi = image_size)

            # The figure will remain open, using memory, unless explicitly closed
            pyplot.close('all')
