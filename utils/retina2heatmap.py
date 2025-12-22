
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

class WaveletHeatmap:
    def __init__(self):
        """
        Generate the heatmap of important structures of the image. 
        The input is the retina image and the output is the heatmap. 
        The method is based on the Wavelet transformation, Sobel filter and morphological operations.
        """
        pass

    def _generate_black_mask(self, img):
        """
        Generate black mask to cover non-retina area.
        """
        black_mask = np.where(img == 0, 1, 0).astype("uint8")
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(black_mask, kernel, iterations=5)

    def _apply_wavelet_transform(self, img):
        """
        Apply Wavelet transformation with the haar base.
        """
        coeffs2 = pywt.dwt2(img, 'haar')
        LL, (LH, HL, HH) = coeffs2
        heatmap = 0.6 * np.abs(LL) + 0.1 * np.abs(LH) + 0.1 * np.abs(HL) + 0.1 * np.abs(HH)
        return cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    def _apply_sobel_filter(self, heatmap):
        """
        Apply Sobel filter and normalise the result.
        """
        grad_x = cv2.Sobel(heatmap.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heatmap.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _normalize_heatmap(self, heatmap):
        """
        Normalize the heatmap to the range [0, 255].
        """
        heatmap = heatmap.astype("float")
        heatmap -= np.min(heatmap)
        heatmap /= np.max(heatmap)
        heatmap *= 255  
        return heatmap.astype("uint8")

    def generate_heatmap(self, img):
        """
        Generate the heatmap.

        :param img: Input opencv2 RGB image.
        :return: Normalized heatmap in range [0, 255] (greyscale).
        """
        black_mask = self._generate_black_mask(img)
        heatmap = self._apply_wavelet_transform(img)
        heatmap = self._apply_sobel_filter(heatmap)
        heatmap = np.where(heatmap > np.quantile(heatmap, 0.9), heatmap, 0)

        black_mask = cv2.resize(black_mask, (heatmap.shape[1], heatmap.shape[0]))
        heatmap = np.where(black_mask == 1, 0, heatmap)
        
        return self._normalize_heatmap(heatmap)

    def generate_overlay(self, img_color, img_grey):
        """
        Generate an overlay of the image by the given heatmap.

        :param img_color: retina image.
        :param img_grey: heatmap.
        """
        heatmap = self.generate_heatmap(img_grey)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.resize(heatmap_colored, (img_color.shape[1], img_color.shape[0]))
        return cv2.addWeighted(img_color, 0.5, heatmap_colored, 0.5, 0)

    def show_overlay(self, overlay):
        """
        Show the given overlay.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        