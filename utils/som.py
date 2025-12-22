
import numpy as np
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, width, height, input_dim, learning_rate=0.99, sigma=0.99, max_iter=10_000):
        """
        The Kohonen SOM method.
        :param width: width of the graph.
        :param height: height of the graph.
        :param input_dim: input dimensionality of the feature.
        :param learning_rate:
        :param sigma:
        :param max_iter:
        """

        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.sigma = sigma
        
        self.weights = np.random.rand(height, width, input_dim)
        
        self.neuron_coords = np.array([(i, j) for i in range(height) for j in range(width)])
        self.neuron_coords = self.neuron_coords.reshape((height, width, 2))
    
    def _decay_learning_rate(self, iter):
        return self.learning_rate * np.exp(-iter / self.max_iter)
    
    def _decay_sigma(self, iter):
        return self.sigma * np.exp(-iter / self.max_iter)
    
    def _neighborhood_func(self, dist, sigma):
        return np.exp(-dist**2 / (2 * sigma**2))
    
    def train(self, heatmap):
        """
        Train the Kohonen mapping on the given heatmap.

        :param heatmap: NumPy matrix heatmap.
        """

        if heatmap.flatten().sum() > 0:

            data = self.sample_heatmap(heatmap, 1000)

            for iter in range(self.max_iter):
                sample = data[np.random.randint(0, data.shape[0])]
                
                diff = self.weights - sample  # shape (height, width, input_dim)
                dist = np.linalg.norm(diff, axis=2)
                
                bmu_idx = np.unravel_index(np.argmin(dist), dist.shape)
                neuron_dists = np.linalg.norm(self.neuron_coords - bmu_idx, axis=2)
                
                lr = self._decay_learning_rate(iter)
                sigma = self._decay_sigma(iter)
                h = self._neighborhood_func(neuron_dists, sigma)  # shape (height, width)
                h = h[:, :, np.newaxis]

                self.weights += lr * h * (sample - self.weights)
    
    def get_weights(self):
        """
        Get weights - the graph nodes positionings
        """
        return self.weights

    def sample_heatmap(self, heatmap, sample_size):
        """
        Sample the heatmap based on its values. Coordinates with higher values will appear in the SOM more likely. 
        """
        H, W = heatmap.shape
        total_points = H * W
        flat_heatmap = heatmap.flatten()
        
        probabilities = flat_heatmap / flat_heatmap.sum()
        nonzero_count = np.count_nonzero(probabilities)
        replace_flag = sample_size > nonzero_count

        indices = np.random.choice(
            total_points,
            sample_size,
            replace=replace_flag,
            p=probabilities
        )
        
        xs = indices // W
        ys = indices % W
        
        samples = np.column_stack((ys / H, xs / W)).astype(np.float32)
        return samples
        