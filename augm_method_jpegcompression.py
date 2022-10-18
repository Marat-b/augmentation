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

# form image files with 4 channels

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
            mask = old_image[:,:,[3]]
            # print(f'mask.shape={mask.shape}')
            aug = augments[i]
            new_image = aug(image=old_image[:,:,:3])
            new_image = augments2[i](image=new_image)
            # print(f'new_image.shape={new_image.shape}')
            new_image = cv2.merge((new_image, mask))
            if suffix is None:
                imageio.imwrite('{}.{}'.format(join(output_dir, file_name), ext), new_image)
            else:
                imageio.imwrite('{}_{}.{}'.format(join(output_dir, file_name), suffix, ext), new_image)

