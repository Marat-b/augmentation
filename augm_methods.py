import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pathlib

def main(args):
    augments = [
        # iaa.AdditiveGaussianNoise(scale=(10, 60)),
        # iaa.AdditiveGaussianNoise(per_channel=True, scale=40),
        # iaa.Dropout(per_channel=True, p=0.3),
        # iaa.CoarseDropout(p=0.4, size_percent=0.1),
        iaa.SaltAndPepper(p=0.4),
        # iaa.CoarseSaltAndPepper(p=0.4, size_percent=0.2),
        # iaa.JpegCompression(compression=97),
        # iaa.blend_alpha(0.9, iaa.MedianBlur(31)),
        iaa.GaussianBlur(sigma=13.0),
        # iaa.MeanShiftBlur(spatial_radius=20.0),
        # iaa.SigmoidContrast(gain=10, cutoff=0.1),
        # iaa.LinearContrast(alpha=0.45),
        # iaa.Sharpen(alpha=1, lightness=1.0),
        # iaa.Emboss(alpha=1, strength=2.0),
        # iaa.EdgeDetect(alpha=0.35),
        # iaa.imgcorruptlike.Snow(severity=3),
        # iaa.imgcorruptlike.Spatter(severity=5),
        # iaa.FastSnowyLandscape(lightness_multiplier=2.0, lightness_threshold=100),
        # iaa.Clouds(),
        # iaa.Fog(),
        # iaa.Snowflakes(flake_size=(0.8, 0.9), speed=(0.01, 0.5), density=0.075),
        # iaa.Rain(drop_size=0.9)
        iaa.MedianBlur(k=23),
        # iaa.MotionBlur(k=19),
        # iaa.imgcorruptlike.GlassBlur(severity=5, seed=2),
        iaa.Voronoi(iaa.RegularGridPointsSampler(n_cols=100, n_rows=130))
    ]
    # index = 18
    # old_image = imageio.imread('{}{}'.format(path_in, files[i]))
    # old_image = imageio.imread(r'F:\VMWARE\FOLDER\UTILZ\MaskRCNN\potato\dataset\raw\train20220814\scab20220809\20220726_162915.png')
    # cv2_imshow(old_image[:, :, ::-1])

    # new_image = augments[index](image=old_image)
    # ia.imshow(new_image)

    # for i in range(len(augments)):
    #     new_image = augments[i](image=old_image)
    #     ia.imshow(new_image)
    # path_in = '{}{}'.format(root_path, path.format(index, '', ''))
    # print(f'path_in={path_in}')

    path_in =  args.input_directory
    path_out = args.output_directory
    for j in range(len(augments)):
        path_out_new = '{}_a{}'.format(path_out, j)
        # print(f'path_out={path_out}')
        files = [f for f in listdir(path_in) if isfile(join(path_in, f)) and str(f).split('.')[1] != 'db']
        # count_files = len(files)

        # print(f'files={count_files}')
        pathlib.Path(path_out_new).mkdir(parents=True, exist_ok=True)
        for file in tqdm(files):
            # print(f'file={file}')
            old_image = imageio.imread('{}/{}'.format(path_in, file))
            # cv2_imshow(old_image[:, :, ::-1])
            new_image = augments[j](image=old_image)
            # cv2_imshow(new_image[:, :, ::-1])
            imageio.imwrite('{}/{}'.format(path_out_new, file), new_image)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate augmented files")
    parser.add_argument(
        "-id", "--input_dir", dest="input_directory",
        help="input path to a PNG files and output path by default "
        )
    parser.add_argument(
        "-od", "--output_dir", dest="output_directory", default=None,
        help="another path to a augmented PNG files"
        )
    args = parser.parse_args()
    main(args)