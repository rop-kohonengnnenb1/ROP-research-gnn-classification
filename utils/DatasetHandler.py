
import os, re, json
import cv2

import torch
import torch_geometric.data as geo_data
import torchvision.transforms as transforms
import numpy as np

from torch_geometric.utils import grid
from collections import Counter


class GraphImageDataset(geo_data.Dataset):
    def __init__(self, graph_dir, image_dir, patches_dir, bin_ids):
        super().__init__()
        """
        Graph dataset object.

        :param graph_dir: Path to the folder with the dataset graphs defining patches (predefined based on Kohonen mapping). 
        :param image_dir: Path to the folder with the dataset images.
        :param patches_dir: Path to the folder with the dataset images patches (predefined based on Kohonen mapping). 
        :param bin_ids: List of lists of integer bin IDs to check (e.g., [0, 1, 3]). Defines the training, validation and testing partitions.
        """
        self.graph_dir = graph_dir
        self.image_dir = image_dir
        self.patches_dir = patches_dir

        self.graph_files = [
            f for f in os.listdir(graph_dir) if f.endswith('npy')
            and self.is_image_in_bins(bin_ids, f)
        ]
        
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if not f.startswith('.') and f.lower().endswith(('jpg', 'png', 'jpeg'))
            and self.is_image_in_bins(bin_ids, f)
        ]

        self.patch_files = [
            f for f in os.listdir(patches_dir) 
            if not f.startswith('.') and f.lower().endswith('npy')
            and self.is_image_in_bins(bin_ids, f)
        ]
        
        self.dg_pattern = re.compile(r'_DG(\d+)_')
        self.label_frequencies = self._count_label_frequencies()
        self.unique_labels = sorted(self.label_frequencies.keys())
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(self.unique_labels))}
    
    def __len__(self):
        return len(self.graph_files)

    def is_image_in_bins(self, bin_ids, image_name, json_path):
        """
        Checks whether the given image name is present in any of the specified bin IDs.

        :param bin_ids: List of integer bin IDs to check (e.g., [0, 1, 3]).
        :param image_name: Name of the image (without extension) to look for.
        :param json_path: Path to the JSON file with bin assignments (bin ID -> sample name assignment).
        :return: True if the image is in one of the bins, False otherwise.
        """
        image_name = os.path.splitext(image_name)[0]

        with open(json_path, "r") as f:
            bins = json.load(f)

        for bin_id in bin_ids:
            images = bins.get(str(bin_id)) or bins.get(bin_id)
            if images and image_name in images:
                return True

        return False

    def _count_label_frequencies(self):
        """
        Compute frequencies of particular classes labels.

        :return: Frequencies dictionary (label -> frequency).
        """
        dg_counts = Counter()
        for filename in self.image_files:
            match = self.dg_pattern.search(filename)
            if match:
                dg_value = int(match.group(1))
                dg_counts[dg_value] += 1
            else:
                raise ValueError(
                    f"Error: File {filename} does not contain a valid DG label."
                )
        return dg_counts

    def get_class_weights(self):
        """
        Based on classes labels frequencies compute the classes weights (imbalanced dataset)
        
        :return: Weights dictionary (label -> frequency).
        """
        total_images = sum(self.label_frequencies.values())
        class_weights = {
            label: total_images / count 
            for label, count in self.label_frequencies.items()
        }
        total_weight = sum(class_weights.values())
        normalized_weights = {
            label: weight / total_weight 
            for label, weight in class_weights.items()
        }
        weights_tensor = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for label in self.label_to_index:
            idx = self.label_to_index[label]
            weights_tensor[idx] = normalized_weights[label]
        return weights_tensor
    
    def __getitem__(self, idx):
        """
        Generated a torch geometric sample object. 

        :param idx: Index of the particular sample in the dataset.
        :return: torch geometric sample object (geo_data.Data).
        """
    
        graph_file = self.graph_files[idx]
        image_file = graph_file.replace('.npy', '.jpg')  # Use '.jpg' extension for the image

        # Load patches
        patches_path = os.path.join(self.patches_dir, graph_file)
        patches = torch.from_numpy(np.load(patches_path).astype(np.float32))
        # Shape: (12*12, 3, 100, 100)
        
        # Create grid-based edge_index for a 12x12 grid
        edge_index, _ = grid(height=12, width=12)
        # edge_index has shape: (2, 144 * 2) because it's for a grid of 144 graphs (12x12)

        # Load the image
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.resize(
            cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (360, 360)
        )

        # Create the transformation for the PIL image
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0))  # Add an additional batch dimension, resulting in (1, 3, 360, 360)
        ])

        # Apply the transformation to the image
        transformed_image = transform(image)
        # transformed_image has shape: (1, 3, 360, 360)

        match = self.dg_pattern.search(image_file)
        if not match:
            raise ValueError(
                f"Error: File {image_file} does not contain a valid DG label."
            )
        label = int(match.group(1))
        label = self.label_to_index[label]

        return geo_data.Data(
            image=transformed_image, 
            x=patches,
            edge_index=edge_index, 
            label=label
        )
