import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path


class AugmenForeground():
    """
    Augmentation of foreground pictures
    """

    def __init__(self, out_dir=None):
        self.out_dir = out_dir

    def make_augmentation(self, dir_path):
        # files = [entry for entry in os.scandir(dir_path) if entry.is_file()]
        files = [entry for entry in Path(dir_path).iterdir() if entry.is_file() and str(entry).split('.')[1] != 'db']
        for file in tqdm(files):
            # print(f'file={file}')
            self.fliplr(file)
            self.flipud(file)
            # self.affine(file)
            # self.scalex(file)
            # self.scaley(file)
            # self.piecewiseaffine(file)
            self.rot90(file)
            self.fliplr_rot90_1(file)
            self.fliplr_rot90_2(file)
            self.fliplr_rot90_3(file)
            self.flipud_rot90_1(file)
            self.flipud_rot90_2(file)
            self.flipud_rot90_3(file)
            # self.scalex_rot90(file)
            # self.scaley_rot90(file)
            # self.scalex_fliplr_rot90(file)
            # self.scaley_fliplr_rot90(file)
            # self.scalex_flipud_rot90(file)
            # self.scaley_flipud_rot90(file)
            self.rotate(file)
            self.fliplr_rotate(file)
            self.flipud_rotate(file)
            # self.scalex_rotate(file)
            # self.scaley_rotate(file)
            # self.scalex_fliplr_rotate(file)
            # self.scalex_flipud_rotate(file)
            # self.scaley_fliplr_rotate(file)
            # self.scaley_flipud_rotate(file)

    def aug_action(self, aug, file_path, new_file_name):
        """
        Augmentation prepare
        :param aug:
        :param file_path:
        :param new_file_name:
        :return:
        """
        image = imageio.imread(file_path)
        image_aug = aug(image=image)
        image_rgb = self.to_tgb(image_aug)
        augmented_new_file_name = 'a_{}'.format(new_file_name)
        self.save_file(image_rgb, file_path, augmented_new_file_name)

    def fliplr(self, file_path: str):
        """
        Flip image to horizontal
        :param file_path:
        :return:
        """
        aug = iaa.Fliplr()
        self.aug_action(aug, file_path, 'fliplr')

    def flipud(self, file_path: str):
        aug = iaa.Flipud()
        self.aug_action(aug, file_path, 'flipud')

    def affine(self, file_path: str):
        aug = iaa.Affine(shear=(-45, 45))
        self.aug_action(aug, file_path, 'affine')

    def scalex(self, file_path: str):
        aug = iaa.Affine(scale={'x': (0.1, 0.5)})
        self.aug_action(aug, file_path, 'scalex')

    def scaley(self, file_path: str):
        aug = iaa.Affine(scale={'y': (0.1, 0.5)})
        self.aug_action(aug, file_path, 'scaley')

    def piecewiseaffine(self, file_path: str):
        aug = iaa.PiecewiseAffine(scale=(0.01, 0.4))
        self.aug_action(aug, file_path, 'piecewiseaffine')

    def rotate(self, file_path: str):
        aug = iaa.Affine(rotate=(45))
        self.aug_action(aug, file_path, 'rotate45')
        aug = iaa.Affine(rotate=(135))
        self.aug_action(aug, file_path, 'rotate135')
        aug = iaa.Affine(rotate=(225))
        self.aug_action(aug, file_path, 'rotate225')
        aug = iaa.Affine(rotate=(315))
        self.aug_action(aug, file_path, 'rotate315')

    def rot90(self, file_path: str):
        aug = iaa.Rot90((1, 1))
        self.aug_action(aug, file_path, 'rot90_1')
        aug = iaa.Rot90((2, 2))
        self.aug_action(aug, file_path, 'rot90_2')
        aug = iaa.Rot90((3, 3))
        self.aug_action(aug, file_path, 'rot90_3')

    def fliplr_rot90_1(self, file_path: str):
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'fliplr_rot90_1')

    def fliplr_rot90_2(self, file_path: str):
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'fliplr_rot90_2')

    def fliplr_rot90_3(self, file_path: str):
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'fliplr_rot90_3')

    def flipud_rot90_1(self, file_path: str):
        aug = iaa.Sequential([iaa.Flipud(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'flipud_rot90_1')

    def flipud_rot90_2(self, file_path: str):
        aug = iaa.Sequential([iaa.Flipud(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'flipud_rot90_2')

    def flipud_rot90_3(self, file_path: str):
        aug = iaa.Sequential([iaa.Flipud(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'flipud_rot90_3')

    def scalex_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scalex_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scalex_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scalex_rot90_3')

    def scalex_fliplr_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rot90_3')

    def scalex_flipud_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scalex_flipud_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scalex_flipud_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scalex_flipud_rot90_3')

    def scaley_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scalex_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scalex_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scalex_rot90_3')

    def scaley_fliplr_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rot90_3')

    def scaley_flipud_rot90(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((1, 1))])
        self.aug_action(aug, file_path, 'scaley_flipud_rot90_1')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((2, 2))])
        self.aug_action(aug, file_path, 'scaley_flipud_rot90_2')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Rot90((3, 3))])
        self.aug_action(aug, file_path, 'scaley_flipud_rot90_3')

    ################## rotate ####################################################
    def fliplr_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'fliplr_rotate_45')
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'fliplr_rotate_135')
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'fliplr_rotate_225')
        aug = iaa.Sequential([iaa.Fliplr(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'fliplr_rotate_315')

    def flipud_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Flipud(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'flipud_rotate_45')
        aug = iaa.Sequential([iaa.Flipud(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'flipud_rotate_135')
        aug = iaa.Sequential([iaa.Flipud(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'flipud_rotate_225')
        aug = iaa.Sequential([iaa.Flipud(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'flipud_rotate_315')

    def scalex_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scalex_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scalex_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scalex_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scalex_rotate315')

    def scaley_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scaley_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scaley_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scaley_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scaley_rotate315')

    def scalex_fliplr_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scalex_fliplr_rotate315')

    def scalex_flipud_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scalex_flipud_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scalex_flipud_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scalex_flipud_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'x': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scalex_flipud_rotate315')

    def scaley_fliplr_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Fliplr(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scaley_fliplr_rotate315')

    def scaley_flipud_rotate(self, file_path: str):
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(45))])
        self.aug_action(aug, file_path, 'scaley_flipud_rotate45')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(135))])
        self.aug_action(aug, file_path, 'scaley_flipud_rotate135')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(225))])
        self.aug_action(aug, file_path, 'scaley_flipud_rotate225')
        aug = iaa.Sequential([iaa.Affine(scale={'y': (0.1, 0.5)}), iaa.Flipud(), iaa.Affine(rotate=(315))])
        self.aug_action(aug, file_path, 'scaley_flipud_rotate315')

    ################################################################################

    def to_tgb(self, old_image):
        """
        Translate from BGR to RGB
        :param old_image:
        :return: file in RGB format
        """
        new_image = np.ones(old_image.shape, old_image.dtype)
        new_image[:, :, 0] = old_image[:, :, 2]
        new_image[:, :, 1] = old_image[:, :, 1]
        new_image[:, :, 2] = old_image[:, :, 0]
        new_image[:, :, 3] = old_image[:, :, 3]
        return new_image

    def save_file(self, image, file_path, new_name):
        """
        Save new RGB file
        :param new_name: new name of file
        :param image: file in RGB format
        :param file_path: path of new file
        :return:
        """
        file_template = '{}/{}_{}.png'
        if self.out_dir is None:
            dirname = os.path.dirname(file_path)
        else:
            dirname = self.out_dir
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file = file_template.format(dirname, base_name, new_name)
        cv2.imwrite(new_file, image)

    def main(self, args):
        self.out_dir = args.output_directory
        self.make_augmentation(args.input_directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate augmented files")
    parser.add_argument("-id", "--input_dir", dest="input_directory",
                        help="input path to a PNG files and output path by default ")
    parser.add_argument("-od", "--output_dir", dest="output_directory", default=None,
                        help="another path to a augmented PNG files")
    args = parser.parse_args()
    af = AugmenForeground()
    af.main(args)
    # af.make_augmentation(IMAGES_PATH)
    # IMAGES_PATH = 'Y:\\UTILZ\\canteen\\set5\\input\\foregrounds\\bread\\white_bread'
    # IMAGES_PATH = 'Y:\\UTILZ\\canteen\\set6'
    # IMAGES_PATH = r'Y:\UTILZ\MaskRCNN\potato\set2\input\foregrounds\potato\potato_strong'
    # image = imageio.imread(IMAGES_PATH + "bread.png")
    # bn = os.path.basename(IMAGES_PATH)
    # print(os.path.basename(IMAGES_PATH))
    # print(os.path.dirname(IMAGES_PATH))
    # print(os.path.splitext(bn)[0])
