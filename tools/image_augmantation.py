import imgaug as ia
import imageio
import imgaug.augmenters as iaa
import glob
import os

# Input path
dataset_for_aug = glob.glob("aug/*.jpg")

images = []
images_aug = []

for path in dataset_for_aug:

    image = imageio.imread(path)

    images.append(image)


seq = iaa.Sequential(
    [
        iaa.SomeOf((3, 4),
                   [
                       iaa.Add((-40, 40), per_channel=0.5),
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       iaa.Multiply((0.5, 1)),
                       iaa.JpegCompression(compression=(70, 99)),
                       iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100))),
                       iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
                                                  foreground=iaa.AddToHue((-100, 100))),
                       iaa.GaussianBlur(sigma=(0.0, 1.0)),
                       iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                       iaa.GammaContrast((0.5, 2.0), per_channel=True),
                       iaa.Affine(rotate=(-15, 15)),
                       iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255)),
                       iaa.Affine(translate_percent={"y": -0.20}, mode=ia.ALL, cval=(0, 255)),
                       iaa.Affine(translate_percent={"x": +0.20}, mode=ia.ALL, cval=(0, 255)),
                       iaa.Affine(translate_percent={"y": +0.20}, mode=ia.ALL, cval=(0, 255)),

                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

images_aug = seq(images=images)

i = 0
for img in images_aug:
    imageio.imwrite(os.path.basename(dataset_for_aug[i]), img)
    i += 1
