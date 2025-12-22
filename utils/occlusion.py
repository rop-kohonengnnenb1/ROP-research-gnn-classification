
import torch
import numpy as np

from tqdm import tqdm


class OcclusionExplainer:
    def __init__(self, model, device='cpu'):
        """
        Occlusion method (perturbation-based explainer).
        :param model: input model.
        :param device: cpu/cuda device.
        """
        self.model = model
        self.device = device

    def explain(self, image_tensor, class_idx, patch_size=50, stride=5):
        """
        Explain the model decision and create the importance heatmap.

        :param image_tensor: Input image torch.Tensor [1, C, H, W].
        :param class_idx: Integer - class index for which we create the explaination heatmap.
        :param patch_size: Size of the explain window (it proposed the same size as the size of the patches in GNN, but is is not necessary).
        :param stride: Stride of the explain window.
        :return: Importance heatmap [H, W] as NumPy array normalized between -1 and 1 (higher value means higher importance of particular pixel / area). 
        """

        image_tensor = image_tensor.to(self.device)
        _, C, H, W = image_tensor.shape

        # Get base score (without occlusion)
        with torch.no_grad():
            input_np = torch.permute(image_tensor.squeeze(0), (1, 2, 0)).cpu().numpy()
            output_probs = self.model.forward(input_np).squeeze(0)  # (num_classes,)
            base_score = float(output_probs[class_idx])

        importance_map = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        for y in tqdm(range(0, H - patch_size + 1, stride), desc="Occlusion sliding"):
            for x in range(0, W - patch_size + 1, stride):
                occluded = image_tensor.clone()

                # Set occlusion (zeros)
                occluded[0, :, y:y+patch_size, x:x+patch_size] = 0

                with torch.no_grad():
                    input_np = torch.permute(occluded.squeeze(0), (1, 2, 0)).cpu().numpy()
                    output_probs = self.model.forward(input_np).squeeze(0)
                    score = float(output_probs[class_idx])

                delta = base_score - score  # Score change based on the area occlusion.

                # Add score change to the particular area on the heatmap.
                importance_map[y:y+patch_size, x:x+patch_size] += delta
                counts[y:y+patch_size, x:x+patch_size] += 1

        counts[counts == 0] = 1
        importance_map /= counts

        # Normalize the heatmap to the range [-1, 1]
        max_abs = np.max(np.abs(importance_map)) + 1e-8
        importance_map = importance_map / max_abs

        return importance_map
