import cv2
import skimage.measure
from skimage.filters import sobel
import SimpleITK as sitk
from pybdm import BDM
import numpy as np
from PIL import Image
from cv2 import imread
from radiomics import featureextractor
import traceback
import typing
import sys
import attr

# The cpbd module tries to import scipy.ndimage.imread, which does not exist in current versions of scipy. The statement below is a workaround for this issue.
# noinspection PyTypeChecker
sys.modules['scipy.ndimage.imread'] = cv2.imread
import cpbd

# Define lists of bins for the function that calculates the K complexity feature
# TODO: make the binning user configurable
bins_0_252 = list(range(0, 252, 28))
bins_0_0_9 = list(np.arange(0, 0.9, 0.1))

full_feature_list = [
    'contrast_rms',
    'contrast_tenengrad',
    'fractal_dimension',
    'sharpness',
    'sharpness_laplacian',
    'colorfulness',
    'pixel_intensity_mean',
    'hue_mean',
    'saturation_mean',
    'k_complexity',
    'entropy_shannon',
    'pyradiomics_features'
]


@attr.s
class FeatureExtractor:
    str_iterable_validator = attr.validators.instance_of(typing.Iterable[str])
    # TODO: Expose all pyradiomics settings to pass through
    pyradiomics_config: dict = {'voxelBatch': 100}
    enabled_features = attr.ib(default='all')
    other_config: dict = {
        'number_of_fractal_dimension_scales': 10,
        'extract_lab_channels': True,
        'extract_rgb_channels': True
    }
    auto_color_convert: typing.Type[int] = True
    # TODO: allow finegrained enabling of pyradiomics features, instead of by class
    pyradiomics_feature_classes: str_iterable_validator = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    extracted_features: dict = {}

    reuse_loaded_image = False
    _cv2_img = None
    _cv2_img_bw = None
    _cv2_img_hsv = None
    _cv2_img_lab = None
    _PIL_img = None
    _PIL_img_bw = None
    img_path = None
    fractal_dimension_scales = np.logspace(
        start=0.01,
        stop=1,
        num=other_config['number_of_fractal_dimension_scales'],
        endpoint=False,
        base=2)

    # Initialize and configure Pyradiomics feature extractor
    pyradiomics_extractor = featureextractor.RadiomicsFeatureExtractor()
    pyradiomics_extractor.settings.update(pyradiomics_config)
    pyradiomics_extractor.disableAllFeatures()
    for feature_class in pyradiomics_feature_classes:
        pyradiomics_extractor.enableFeatureClassByName(feature_class)

    @property
    def enabled_features_list(self):
        if self.enabled_features == 'all' or self.enabled_features is None:
            self.enabled_features = full_feature_list
        return self.enabled_features

    @property
    def PIL_img(self):
        if self._PIL_img is None:
            try:
                self._PIL_img = Image.open(self.img_path)
                return self._PIL_img
            except (OSError, IOError):
                print(f'Opening image at {self.img_path} failed: \n {traceback.format_exc()}')
        else:
            return self._PIL_img

    @property
    def PIL_img_bw(self):
        if self._PIL_img_bw is None:
            self._PIL_img_bw = self.PIL_img.convert('L')
        return self._PIL_img_bw

    @property
    def cv2_img(self):
        if self._cv2_img is None:
            try:
                self._cv2_img = imread(self.img_path)
                return self._cv2_img
            except (OSError, IOError):
                print(f'Opening image at {self.img_path} failed: \n {traceback.format_exc()}')
        else:
            return self._cv2_img

    @property
    def cv2_img_bw(self):
        if self._cv2_img_bw is None:
            self._cv2_img_bw = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2GRAY)
        return self._cv2_img_bw

    @property
    def cv2_img_hsv(self):
        if self._cv2_img_hsv is None:
            self._cv2_img_hsv = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2HSV)
        return self._cv2_img_hsv

    @property
    def cv2_img_lab(self):
        if self._cv2_img_lab is None:
            self._cv2_img_lab = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2Lab)
        return self._cv2_img_lab

    def extract(self, img_path):
        self.extracted_features = {}
        self.img_path = img_path
        self.extracted_features.update({'img_path': self.img_path})
        extractor_mapping = {
            'contrast_rms': self.contrast_rms,
            'contrast_tenengrad': self.contrast_tenengrad,
            'fractal_dimension': self.fractal_dimension,
            'sharpness': self.sharpness,
            'sharpness_laplacian': self.sharpness_laplacian,
            'colorfulness': self.colorfulness,
            'pixel_intensity_mean': self.pixel_intensity_mean,
            'saturation_mean': self.saturation_mean,
            'hue_mean': self.hue_mean,
            'entropy_shannon': self.entropy_shannon,
            'k_complexity': self.k_complexity,
            'pyradiomics_features': self.pyradiomics_features
        }
        # Call all enabled feature extractors
        for extractor in extractor_mapping:
            if extractor in self.enabled_features_list:
                extractor_mapping[extractor]()
        out = self.extracted_features

        # Reset state
        self._cv2_img = None
        self._cv2_img_bw = None
        self._cv2_img_hsv = None
        self._cv2_img_lab = None
        self._PIL_img = None
        self._PIL_img_bw = None
        self.img_path = None
        return self.extracted_features

    def contrast_rms(self):
        feature_value = self.cv2_img_bw.std()
        self.extracted_features.update({'contrast_rms': feature_value})

    def contrast_tenengrad(self):
        sobel_img = sobel(self.cv2_img_bw) ** 2
        feature_value = np.sqrt(np.sum(sobel_img)) / self.cv2_img_bw.size * 10000
        self.extracted_features.update({'contrast_tenengrad': feature_value})

    def fractal_dimension(self):
        # Adapted from https://francescoturci.net/2016/03/31/box-counting-in-numpy/
        # Find all the non-zero pixels
        pixels = []
        for i in range(self.cv2_img_bw.shape[0]):
            for j in range(self.cv2_img_bw.shape[1]):
                if self.cv2_img_bw[i, j] > 0:
                    pixels.append((i, j))

        lx = self.cv2_img_bw.shape[1]
        ly = self.cv2_img_bw.shape[0]
        pixels = np.array(pixels)

        # Compute the fractal dimension considering only scales in a logarithmic list
        ns = []
        # Loop over the scales
        for scale in self.fractal_dimension_scales:
            # Compute the histogram
            h, edges = np.histogramdd(pixels, bins=(np.arange(0, lx, scale), np.arange(0, ly, scale)))
            ns.append(np.sum(h > 0))

        # linear fit, polynomial of degree 1
        coeffs = np.polyfit(np.log(self.fractal_dimension_scales), np.log(ns), 1)
        feature_value = -coeffs[0]  # the fractal dimension is the OPPOSITE of the fitting coefficient
        self.extracted_features.update({'contrast_rms': feature_value})

    def sharpness(self):
        feature_value = cpbd.compute(self.cv2_img_bw)
        self.extracted_features.update({'sharpness': feature_value})

    def sharpness_laplacian(self):
        feature_value = cv2.Laplacian(self.cv2_img_bw, cv2.CV_64F).var()
        self.extracted_features.update({'sharpness_laplacian': feature_value})

    def colorfulness(self):
        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        # "Measuring colourfulness in natural images" David Hasler and Sabine Susstrunk
        (B, G, R) = cv2.split(self.cv2_img.astype('float'))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        feature_value = std_root + (0.3 * mean_root)
        self.extracted_features.update({'colorfulness': feature_value})

    def pixel_intensity_mean(self):
        feature_value = self.cv2_img.mean()
        self.extracted_features.update({'pixel_intensity_mean': feature_value})

    def hue_mean(self):
        feature_value = self.cv2_img_hsv.mean()
        self.extracted_features.update({'hue_mean': feature_value})

    def saturation_mean(self):
        feature_value = self.cv2_img_hsv[:, :, 1].mean()
        self.extracted_features.update({'saturation_mean': feature_value})

    def entropy_shannon(self):
        feature_value = skimage.measure.shannon_entropy(self.cv2_img)
        self.extracted_features.update({'entropy_shannon': feature_value})

    def k_complexity(self):
        image = self.cv2_img_bw
        image = image.reshape(image.shape[0] * image.shape[1])
        image = image * (252 / 256)
        image = np.digitize(image, bins=bins_0_252) - 1
        bdm = BDM(ndim=1, nsymbols=9, warn_if_missing_ctm=False)
        feature_value = bdm.bdm(image)
        self.extracted_features.update({'k_complexity_bw': feature_value})

        if self.other_config['extract_lab_channels']:
            for k, v in {'L': 0, 'a': 1, 'b': 2}.items():
                image_ = self.cv2_img_lab * (252 / 256)
                image_ = image_.astype('float32') / 255  # transformation to fix Euclidian distances in Lab space
                image_ = image_[:, :, v]
                image_ = image_.reshape(image_.shape[0] * image_.shape[1])
                image_ = np.digitize(image_, bins=bins_0_0_9) - 1
                feature_value = bdm.bdm(image_)
                self.extracted_features.update({f'k_complexity_{k}_channel': feature_value})

    def pyradiomics_features(self):
        img = self.PIL_img
        img = np.array(img)
        im = sitk.GetImageFromArray(img)

        # Create dummy mask that exposes the whole image
        ma = sitk.GetImageFromArray(np.ones(img.shape, dtype='uint8'))
        ma.CopyInformation(im)

        # Extract features
        features = self.pyradiomics_extractor.execute(im, ma, label=1)

        # Flatten and tidy up the dictionary
        stats_keys = [x for x in list(features.keys()) if x[0][0] == 'o']
        stats_dict = {k[9:]: float(features[k]) for k in stats_keys}
        self.extracted_features.update(stats_dict)
