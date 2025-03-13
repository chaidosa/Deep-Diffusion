# KPConvEncoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .KPConv import KPConv
from loguru import logger

class KPConvEncoder(nn.Module):
    """
    KPConv-based encoder.
    It takes a point cloud of shape [B, N, 3] (or with extra features concatenated)
    and produces a global latent vector split into mean (mu) and sigma (log–scale),
    each of shape [B, zdim].
    """
    def __init__(self, zdim, input_dim, extra_feature_channels=0, kpconv_configs=None):
        super().__init__()
        # Define the input channel count.
        # If you have extra features, they should be concatenated with the 3D coordinates.
        in_channels = extra_feature_channels + input_dim
        # kpconv_configs is a list of dictionaries; if not provided, we use default settings.
        # Each dictionary defines one set–abstraction layer.
        if kpconv_configs is None:
            kpconv_configs = [
                {"num_kernel_points": 15, "out_channels": 32, "radius": 0.1, "kp_influence": "linear"},
                {"num_kernel_points": 15, "out_channels": 64, "radius": 0.2, "kp_influence": "linear"}
            ]
        self.kpconv_layers = nn.ModuleList()
        # We will use the query points themselves for grouping (i.e. no subsampling).
        for cfg in kpconv_configs:
            layer = KPConv(in_channels, 
                           cfg["out_channels"],
                           num_kernel_points=cfg["num_kernel_points"],
                           radius=cfg["radius"],
                           kp_influence=cfg["kp_influence"],
                           fixed_kernel_points="center")
            self.kpconv_layers.append(layer)
            in_channels = cfg["out_channels"]
        # Global pooling: we use a max–pool over the point dimension.
        # Final MLP: projects from the last feature dimension to 2*zdim (mu and sigma).
        self.fc = nn.Linear(in_channels, zdim * 2)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.zdim = zdim

    def forward(self, x):
        """
        Args:
          x: [B, N, 3] point cloud (or with extra channels, e.g., [B, N, 6]).
        Returns:
          A dict with keys:
            "mu_1d": [B, zdim] latent mean
            "sigma_1d": [B, zdim] latent sigma (or log–sigma)
        """
        # For simplicity, we assume the initial features are the raw coordinates.
        # If x has more than 3 channels, you may want to separate them.
        query_xyz = x  # [B, N, 3]
        # For the first layer, use x as features.
        features = x
        # for kpconv in self.kpconv_layers:
        #     query_xyz, features = kpconv(query_xyz, features)
        for kpconv in self.kpconv_layers:
            query_xyz, features = kpconv(query_xyz, features)
            logger.info("After KPConv layer: min={:.5f}, max={:.5f}, mean={:.5f}", features.min().item(), features.max().item(), features.mean().item())
            features = F.relu(features)

        # Global max pooling: reduce over the N dimension.
        global_feat, _ = features.max(dim=1)  # [B, F]
        out = self.fc(global_feat)  # [B, 2*zdim]
        mu, sigma = out[:, :self.zdim], out[:, self.zdim:]
        return {"mu_1d": mu, "sigma_1d": sigma}
