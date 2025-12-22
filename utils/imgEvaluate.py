
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.data as geo_data
import torchvision.transforms as transforms

from torch_geometric.utils import grid


class imgEvaluate:
    def __init__(self, model, device=None):
        """
        Evaluator of single images. Performs evaluation of a single RGB opencv image and returns class logits.

        :param model: Model to be evaluated (expects KohonenGNNenb1).
        :param device: cpu/cuda device.
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

        self.model = model
        self.device = device
    
    def _get_patches_torch(self, transformed_image1200x1200_torch, graph_pos, patch_size=100):
        """
        Get patches based on the input image and graph nodes positionings.

        :param transformed_image1200x1200_torch: 1200x1200 input pytorch tensor image.
        :param graph_pos: Graph nodes positionings.
        :param patch_size: Size (patch_size x patch_size) of each patch.
        :return: Pytorch tensor of patches for the given input.
        """
        patches = []
        half_s = patch_size // 2
        padded_image = F.pad(transformed_image1200x1200_torch, 
            (patch_size * 2, patch_size * 2,
            patch_size * 2, patch_size * 2), 
            mode='constant', value=0
        )
        for pos in graph_pos:
            row, col = int(pos[0]) + patch_size, int(pos[1]) + patch_size
            patch = padded_image[:, row - half_s: row + half_s, col - half_s: col + half_s]
            patches.append(patch)
        return torch.stack(patches)

    def _prepare_data(self, rgb_image_cv2, graph_pos_np, device):
        """
        Prepare input data (preprocessing, resize and transforms).

        :param rgb_image_cv2: Input RGB opencv image.
        :param graph_pos_np: Input NumPy graph nodes positionings tensor.
        :param device: cpu/cuda device.
        :return: geo_data.Data object with sample (image = whole input image, x = patches, edge_index = graph topology, batch = batch mapping).
        """
        image360x360_cv2 = cv2.resize(rgb_image_cv2, (360, 360))
        image1200x1200_cv2 = cv2.resize(rgb_image_cv2, (1200, 1200))

        transformed_image360x360_torch = self.transform(image360x360_cv2)
        transformed_image1200x1200_torch = self.transform(image1200x1200_cv2)

        graph_pos_torch = torch.tensor(graph_pos_np.astype(np.float32)*1200).view(-1, 2)
        edge_index_torch, _ = grid(height=12, width=12)
        patches_torch = self._get_patches_torch(transformed_image1200x1200_torch[0], graph_pos_torch)

        return geo_data.Data(
            image=transformed_image360x360_torch.to(device), 
            x=patches_torch.to(device),
            edge_index=edge_index_torch.to(device),
            batch=torch.zeros(12*12, dtype=torch.long, device=device)
        )

    def forward(self, rgb_image_cv2, graph_pos_np):
        """
        Perform the model evaluation.

        :param rgb_image_cv2: input opencv RGB image.
        :param graph_pos_np: input NumPy tensor with graph nodes positionings for the particular input image.
        :return: Class labels logits.
        """
        return self.model(
            self._prepare_data(rgb_image_cv2, graph_pos_np, self.device)
        )
