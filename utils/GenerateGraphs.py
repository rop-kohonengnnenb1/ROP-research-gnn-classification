
import os, cv2
import torch
import numpy as np

from tqdm import tqdm

from utils.som import *


class GraphGenerator:
    def __init__(self, input_folder_path, output_folder_path, som_width=12, som_height=12, som_dim=2, resize_shape=(1200, 1200)):
        """
        Based on the predefined heatmaps perform Kohonen mapping and construct the graphs (complex networks) with the defined nodes positionings.

        :param input_folder_path: Path to the heatmaps.
        :param output_folder_path: Path where to store graphs.
        :param som_width: Width of the grid graph.
        :param som_height: Height of the grid graph.
        :param som_dim: Dimensionality of the mapping.
        :resize_shape: Shape of the heatmap on which will be the mapping performed.
        """
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.som_width = som_width
        self.som_height = som_height
        self.som_dim = som_dim
        self.resize_shape = resize_shape

    def process_files(self):
        """
        Process heatmaps and save resulting graphs nodes positionings.
        """
        os.makedirs(self.output_folder_path, exist_ok=True)

        for file in tqdm(os.listdir(self.input_folder_path)):
            output_file = os.path.splitext(file)[0] + ".npy"
            if output_file in os.listdir(self.output_folder_path):
                continue

            heatmap_path = os.path.join(self.input_folder_path, file)
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
            if heatmap is None:
                print(f"Warning: Couldn't read file {heatmap_path}")
                continue
            heatmap = cv2.resize(heatmap, self.resize_shape)

            model = SOM(self.som_width, self.som_height, self.som_dim)
            model.train(heatmap)
            weights = model.get_weights()

            np.save(os.path.join(self.output_folder_path, output_file), weights)


generator = GraphGenerator(
    # PATH TO HEATMAPS 
    # PATH WHERE TO STORE GRAPHS
)
generator.process_files()
