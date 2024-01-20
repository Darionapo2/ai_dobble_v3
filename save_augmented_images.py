from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os, shutil

deck = 'exp0'
data_path = f'new_dataset/{deck}/'

augmented_imgs_path = f'new_dataset/{deck}-augmented2/'

if os.path.exists(augmented_imgs_path):
    shutil.rmtree(augmented_imgs_path)
os.mkdir(augmented_imgs_path)

total_images_to_augment = 50
image_size = 100

width_shift_range = 0.1
height_shift_range = 0.1
brightness_range = [0.2, 1.1]
zoom_range = [0.7, 1.3]

for folder in os.listdir(data_path):
    print(f'[INFO] generating images in folder{folder}')

    for file in os.listdir(f'{data_path}/{folder}'):

        # load each image
        img = load_img(f'{data_path}/{folder}/{file}')
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)

        # create image data augmentation generator
        datagen = ImageDataGenerator(
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            brightness_range = brightness_range,
            zoom_range = zoom_range
        )

        # prepare iterator
        it = datagen.flow(samples, batch_size = 1)

        output_path = f'{augmented_imgs_path}{folder}/'

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

        for i in range(1, total_images_to_augment + 1):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')

            fig = pyplot.figure(frameon = False)
            ax = pyplot.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # plot raw pixel data
            ax.imshow(image)
            saved_filename = f'{output_path}card{folder}' + '_{:03d}.tif'.format(i)
            fig.savefig(fname = saved_filename, dpi = 50)

            # the figure will remain open, using memory, unless explicitly closed
            pyplot.close('all')
