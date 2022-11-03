import pathlib
from os import listdir
from os.path import isfile, join

import cv2
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
import numpy
from tqdm import tqdm

# for image files with 4 channels

def adjust_layers(image):
    blue, green, red, alpha = cv2.split(image)
    # new_alpha = alpha.copy()
    new_alpha = np.where(alpha == 255, alpha, 0)
    # new_alpha = np.where(alpha == 0, alpha, 255)
    # cv2_imshow(new_alpha, 'new_alpha')
    blue = cv2.bitwise_and(blue, blue, mask=new_alpha)
    # cv2_imshow(blue, 'blue')
    blue = np.where(blue > 0, blue, 255)
    # cv2_imshow(blue, 'blue')
    green = cv2.bitwise_and(green, green, mask=new_alpha)
    green = np.where(green > 0, green, 255)
    # cv2_imshow(green, 'green')
    red = cv2.bitwise_and(red, red, mask=new_alpha)
    red = np.where(red > 0, red, 255)
    # cv2_imshow(red, 'red')
    new_image = cv2.merge((blue, green, red, new_alpha))
    # cv2_imshow(new_image[:, :, ::-1], 'new_image')
    # cv2_imshow(new_image, 'new_image')
    return new_image

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate png files")
    parser.add_argument(
        "-id", "--input_dir", dest="input_directory",
        help="input path to a JPG files and output path by default "
    )
    parser.add_argument(
        "-od", "--output_dir", dest="output_directory", default=None,
        help="another path to a resized (changed)  PNG files"
    )
    parser.add_argument(
        "-s", "--suffix", default=None, type=str,
        help="Suffix for name of file"
    )

    args = parser.parse_args()
    input_dir = args.input_directory
    suffix = args.suffix

    augments = [
        iaa.MotionBlur(k=(17,17), angle=90, seed=50)
    ]
    augments2 = [
        iaa.JpegCompression(compression=(90, 90), seed=101)
    ]

    if args.output_directory is None:
        output_dir = args.input_directory
    else:
        output_dir = args.output_directory

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and
             join(input_dir, f).split('.')[1] != 'db']
    count = 0
    for file in tqdm(files):
        # print(file)
        file_name, ext = file.split('.')

        old_image = imageio.imread(join(input_dir, file))
        for i in range(len(augments)):
            # old_image = iaa.Resize(0.5)(image=old_image)
            # old_image = adjust_layers(old_image)
            mask = old_image[:,:,[3]]
            # print(f'mask.shape={mask.shape}')
            aug = augments2[i]
            new_image = aug(image=old_image[:,:,:3])
            # new_image = iaa.Resize(0.5)(image=new_image)
            # new_image = augments2[i](image=new_image)
            # print(f'new_image.shape={new_image.shape}, mask.shape={mask.shape}')
            # new_image = iaa.Resize(4.0)(image=new_image)
            # new_image = cv2.merge((new_image, mask))
            new_image = np.concatenate((new_image, mask), axis=2)
            if suffix is None:
                imageio.imwrite('{}.{}'.format(join(output_dir, file_name), ext), new_image)
            else:
                imageio.imwrite('{}_{}.{}'.format(join(output_dir, file_name), suffix, ext), new_image)

