import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float
from tqdm import trange
import matplotlib.pyplot as plt

def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)

def toeplitizMatrix(n, row):
    vecDD = np.zeros(n)
    vecDD[0], vecDD[1], vecDD[row], vecDD[-1], vecDD[-row] = 4, -1, -1, -1, -1
    return vecDD

def vectorize(matrix):
    return matrix.T.ravel()

def reshape(vector, row, col):
    return vector.reshape((row, col), order='F')

class LIME:
    def __init__(self, iterations=10, alpha=2, rho=2, gamma=0.7, strategy=2):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load_image(self, img):
        """Load and preprocess the input image."""
        self.L = img_as_float(img)

        # Check if the image is grayscale (2D)
        if len(self.L.shape) == 2:  # Grayscale image
            self.L = np.stack((self.L,) * 3, axis=-1)  # Convert to RGB by stacking

        # Check for RGBA (4 channels)
        elif self.L.shape[2] == 4:
            self.L = self.L[:, :, :3]  # Keep only the RGB channels

        self.row, self.col = self.L.shape[:2]
        self.T_hat = np.max(self.L, axis=2)  # Illumination map
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        self.W = self.weighting_strategy()

    def weighting_strategy(self):
        """Compute weights for smooth transitions."""
        dTv, dTh = self.dv @ self.T_hat, self.T_hat @ self.dh
        Wv, Wh = 1 / (np.abs(dTv) + 1), 1 / (np.abs(dTh) + 1)
        return np.vstack([Wv, Wh])

    def illum_map(self):
        """Iteratively solve for the illumination map."""
        T = np.zeros((self.row, self.col))
        G, Z = np.zeros((self.row * 2, self.col)), np.zeros((self.row * 2, self.col))
        u = 1

        for _ in trange(self.iterations):
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

        return T ** self.gamma

    def __T_subproblem(self, G, Z, u):
        """Solve the T subproblem."""
        X = G - Z / u
        Xv, Xh = X[:self.row, :], X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft.fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        return exposure.rescale_intensity(np.real(reshape(T, self.row, self.col)), (0, 1), (0.001, 1))

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __derivative(self, matrix):
        v, h = self.dv @ matrix, matrix @ self.dh
        return np.vstack([v, h])

    def enhance(self, img):
        """Enhance the input image using LIME."""
        self.load_image(img)

        # Check if the input image is 2D (grayscale) or 3D (RGB/RGBA)
        if len(img.shape) == 2:  # Grayscale image
            alpha_channel = None  # No alpha channel for grayscale images
        elif img.shape[2] == 4:  # RGBA image
            alpha_channel = img[:, :, 3]  # Keep the alpha channel
        else:
            alpha_channel = None  # For RGB images

        T = self.illum_map()
        R = self.L / np.repeat(T[:, :, np.newaxis], 3, axis=2)
        
        # Ensure R is in the range [0, 1] and clip values
        R = np.clip(R, 0, 1)  # Clamp the values to avoid issues during conversion

        # Convert R to 8-bit format
        R = img_as_ubyte(R)

        # Combine the enhanced RGB image with the original alpha channel if it exists
        if alpha_channel is not None:
            R = np.dstack((R, alpha_channel))  # Add the alpha channel back in

        return R

def enhance_image(img, output_path=None):
    """Load an image, enhance it, and save the output."""
    #img = io.imread(input_path)
    lime = LIME(iterations=1, alpha=2, rho=2, gamma=0.7, strategy=2)
    enhanced_img = lime.enhance(img)

    # Save and display the enhanced image
    # plt.imsave(output_path, enhanced_img)
    # plt.imshow(enhanced_img)
    # plt.axis('off')
    # plt.show()
    return enhanced_img

# Example usage
if __name__ == '__main__':
    input_image = 'enhanced_face_112_79.png'  # Replace with your input image path
    output_image = 'enhanced_image.png'
    enhance_image(input_image, output_image)
