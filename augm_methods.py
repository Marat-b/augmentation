import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pathlib

augments = [
    # iaa.AdditiveGaussianNoise(scale=(10, 60)),
    # iaa.AdditiveGaussianNoise(per_channel=True, scale=40),
    iaa.Dropout(per_channel=True, p=0.5),
    iaa.CoarseDropout(p=0.4, size_percent=0.1),
    iaa.SaltAndPepper(p=0.4),
    # iaa.CoarseSaltAndPepper(p=0.4, size_percent=0.2),
    iaa.JpegCompression(compression=97),
    # iaa.blend_alpha(0.9, iaa.MedianBlur(31)),
    iaa.GaussianBlur(sigma=9.0),
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
    iaa.MedianBlur(k=15),
    iaa.MotionBlur(k=19),
    iaa.imgcorruptlike.GlassBlur(severity=5, seed=2),
    iaa.Voronoi(iaa.RegularGridPointsSampler(n_cols=90, n_rows=110))
]
index = 18
# old_image = imageio.imread('{}{}'.format(path_in, files[i]))
old_image = imageio.imread(r'C:\softz\work\potato\dataset\set19\00000000.jpg')
# cv2_imshow(old_image[:, :, ::-1])

# new_image = augments[index](image=old_image)
# ia.imshow(new_image)

for i in range(len(augments)):
    new_image = augments[i](image=old_image)
    ia.imshow(new_image)
