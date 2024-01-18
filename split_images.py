import os

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def generate_imgs():

    high_res_path = 'dobble_dataset/new_dataset/pdf_scans/high_resolution'

    for d in os.listdir(high_res_path):
        if d.startswith('exp'):
            path = f'{high_res_path}/{d}'
            i = 0
            for img_filename in os.listdir(path):
                print(img_filename)
                im = Image.open(f'{path}/{img_filename}')
                im_ndarray = np.array(im)

                if img_filename.startswith('scan1'):
                    M = round(im.size[0]/3)
                    N = round(im.size[1]/2)
                else:
                    M = round(im.size[0]/3)
                    N = round(im.size[1]/4)

                tiles = []
                for x in range(0, im_ndarray.shape[0], M):
                    for y in range(0, im_ndarray.shape[1], N):
                        tiles.append(im_ndarray[x: x + M, y: y + N])

                exp = img_filename.split('_')[1].split('.')[0]
                print(exp)
                base_filename = f'card_{exp}'

                for tile in tiles:

                    i += 1
                    plt.imshow(tile, interpolation = 'nearest')
                    plt.axis('off')
                    plt.savefig(f'dobble_dataset/new_dataset/saved_images/{exp}'
                                f'/{i}_{base_filename}.jpg',
                                bbox_inches = 'tight',
                                pad_inches = 0,
                                dpi = 250)



def renumerate(path):
    files = os.listdir(path)
    files.sort(reverse = False, key = lambda x: int(x.split('_')[0]))

    print('sorted_filenames:', files)

    i = 0
    for file in files:
        i += 1
        list_old_filename = file.split('_')
        new_filename = f'{i}_{list_old_filename[1]}_{list_old_filename[2]}'

        old_path = os.path.join(path, file)
        new_path = os.path.join(path, new_filename)

        os.rename(old_path, new_path)

def put_into_folder(path):
    files = os.listdir(path)
    files.sort(reverse = False, key = lambda x: int(x.split('_')[0]))

    for i in range(1, len(files) + 1):
        os.mkdir(os.path.join(path, '{:02d}'.format(i)))

        old_path = os.path.join(path, files[i-1])
        new_path = os.path.join(path, '{:02d}'.format(i)+'/'+f'{files[i-1]}')

        os.rename(old_path, new_path)


def main():
    # generate_imgs()
    # renumerate('dobble_dataset/new_dataset/saved_images/exp0')
    # put_into_folder('new_dataset/saved_images/exp-3')
    pass

if __name__ == '__main__':
    main()