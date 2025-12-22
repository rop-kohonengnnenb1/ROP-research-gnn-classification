
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool

from models.EfficientNet_B1 import EfficientNet_B1


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class GlobalEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

        model_full = EfficientNet_B1(n_classes=11)

        self.backbone = nn.Sequential(
            *list(model_full.features.children())[:-2], 
            nn.Flatten(start_dim=2)
        )
        
        self.dimred = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(1280, out_dim),
            Transpose(1, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.dimred(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, self.out_dim) 
        return x


class FusionClassifier(nn.Module):
    def __init__(self, emb_dim=256, num_classes=11):
        super().__init__()
        self.global_encoder = GlobalEncoder(out_dim=emb_dim)

        self.patches_dimred = nn.Linear(3*30*30, 30*30)

        self.gat1 = GATv2Conv(30*30, emb_dim, heads=4, concat=False)
        self.gat2 = GATv2Conv(2*emb_dim, emb_dim, heads=4, concat=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim//2, num_classes)
        )

    def forward(self, data):
        patches_resized = F.interpolate(
            data.x, size=(30, 30), mode='bilinear', align_corners=False
        )
        patches_flatten = torch.flatten(patches_resized, start_dim=1)
        patches_dimred = self.patches_dimred(patches_flatten)        
        patches_emb = F.elu(self.gat1(patches_dimred, data.edge_index))

        image_resized = F.interpolate(
            data.image, size=(360, 360), mode='bilinear', align_corners=False
        )
        global_feat = self.global_encoder(image_resized)

        local_global_feat = torch.hstack([patches_emb, global_feat])
        local_global_feat_emb = F.elu(self.gat2(local_global_feat, data.edge_index))

        nodes_logits = self.classifier(local_global_feat_emb)
        logits = global_mean_pool(nodes_logits, data.batch)

        return logits
