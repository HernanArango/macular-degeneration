import cv2
import numpy as np


class Retinex:

    def __init__(self):
        self.kernels = []
        self.sigmas = []
        self.weights = []

    @staticmethod
    def odd(x):
        """
        #TODO
        :param x: 
        :return: 
        """
        return x + int(x % 2 == 0)

    @staticmethod
    def cast_uint8(image):
        """
        #TODO
        :param image: 
        :return: 
        """
        return np.uint8(image * (255. - 0.) / (image.max() - image.min()))

    def apply(self, image, kernels=(11,), sigmas=(3,), weights=(1,)):
        """
        #TODO
        :param image: 
        :param kernels: 
        :param sigmas: 
        :param weights: 
        :return: 
        """
        if not isinstance(kernels, (list, tuple)):
            self.kernels = (kernels,)
        else:
            self.kernels = kernels
        if not isinstance(sigmas, (list, tuple)):
            self.sigmas = (sigmas,)
        else:
            self.sigmas = sigmas
        if not isinstance(weights, (list, tuple)):
            self.weights = (weights,)
        else:
            self.weights = weights

        if len(self.kernels) != len(self.sigmas):
            if len(self.kernels) == 1:
                self.kernels = tuple([self.kernels[0]] * len(self.sigmas))
            elif len(self.sigmas) == 1:
                self.sigmas = tuple([self.sigmas[0]] * len(self.kernels))
            else:
                raise ValueError("Parameters mismatch: {} (# kernels) != {} (# sigmas)".format(len(self.kernels),
                                                                                               len(self.sigmas)))

        if len(self.weights) == 1 and len(self.kernels) > 1:
            self.weights = tuple([1. / len(self.kernels)] * len(self.kernels))

        self.kernels = tuple([self.odd(kernel) for kernel in self.kernels])

        low_freqs = [cv2.GaussianBlur(image, ksize=(k, k), sigmaX=s) for k, s in zip(self.kernels, self.sigmas)]
        results = [weight * (np.log1p(image) - np.log1p(low_freq)) for low_freq, weight in zip(low_freqs, self.weights)]

        return self.cast_uint8(sum(results))
