
import os, cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm


class PatchesGenerator:
    def __init__(self, images_folder_path, graphs_folder_path, output_folder_path, resize_size=1200, patch_size=100):
        """
        Generate and save patches based on input images and pregenerated graphs.

        :param images_folder_path: Path to the input images.
        :param graphs_folder_path: Path to the input graphs nodes positionings.
        :param output_folder_path: Path where to save output patches.
        :param resize_size: size of the image which will be used to generate the patches.
        :param patch_size: size (square - size x size) of each patch.
        """
        
        self.images_folder_path = images_folder_path
        self.graphs_folder_path = graphs_folder_path
        self.output_folder_path = output_folder_path
        
        self.resize_size = resize_size
        self.patch_size = patch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])

    def process_files(self):
        """
        Process all images and graphs and save patches files.
        """
        os.makedirs(self.output_folder_path, exist_ok=True)

        for file in tqdm(os.listdir(self.images_folder_path)):
            name = os.path.splitext(file)[0]
            output_file = name + ".npy"

            if output_file in os.listdir(self.output_folder_path):
                continue

            graph_path = os.path.join(self.graphs_folder_path, name + ".npy")
            graph_pos = np.load(graph_path).astype(np.float32) * self.resize_size
            graph_pos = torch.tensor(graph_pos).view(-1, 2)

            image_path = os.path.join(self.images_folder_path, name + ".jpg")
            image = cv2.resize(
                cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (1200, 1200)
            )
            transformed_image = self.transform(image)

            patches = []
            half_s = self.patch_size // 2
            padded_image = F.pad(transformed_image, 
                (self.patch_size * 2, self.patch_size * 2,
                self.patch_size * 2, self.patch_size * 2), 
                mode='constant', value=0
            )
            for pos in graph_pos:
                row, col = int(pos[0]) + self.patch_size, int(pos[1]) + self.patch_size
                patch = padded_image[:, row - half_s: row + half_s, col - half_s: col + half_s]
                patches.append(patch)

            np_patches = torch.stack(patches).numpy()

            np.save(os.path.join(self.output_folder_path, output_file), np_patches)


generator = PatchesGenerator(
    # PATH TO IMAGES
    # PATH TO GRAPHS (NODES POSITIONINGS)
    # PATH WHERE TO STORE PATCHES
)
generator.process_files()
