import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance(x):
    """
    Compute pairwise squared Euclidean distance.
    x: (B, N, 3)
    Returns: (B, N, N)
    """
    B, N, _ = x.shape
    xx = torch.sum(x ** 2, dim=2, keepdim=True)  # (B, N, 1)
    yy = torch.sum(x ** 2, dim=2, keepdim=True).transpose(1, 2)  # (B, 1, N)
    dist = xx + yy - 2 * torch.bmm(x, x.transpose(1, 2))
    return dist

def get_neighbors(x, k):
    """
    For each point in x (B, N, 3), return the indices of its k nearest neighbors.
    """
    # Compute pairwise distances
    dist = pairwise_distance(x)
    # Get top k+1 (including self) then remove self (first neighbor)
    _, idx = torch.topk(-dist, k=k+1, dim=-1)
    return idx[:, :, 1:]  # shape: (B, N, k)

class KPConv(nn.Module):
    """
    A basic implementation of a KPConv layer.
    This layer expects as input:
      • query_points: (B, N, 3)
      • support_points: (B, M, 3)   (often the same as query_points)
      • neighbor_idx: (B, N, k)     (indices into support_points)
      • features: (B, M, in_channels)
    and returns (B, N, out_channels).
    """
    def __init__(self, in_channels, out_channels, kernel_size, radius,
                 fixed_kernel_points='center', influence='linear', aggregation_mode='sum'):
        super(KPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.fixed_kernel_points = fixed_kernel_points
        self.influence = influence
        self.aggregation_mode = aggregation_mode
        # Learnable weight for each kernel point
        # self.weights = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels) * 0.01)
        self.weights = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels) * 0.005)

        # Fixed kernel point positions (set once during initialization)
        self.register_buffer('kernel_points', self.initialize_kernel_points(kernel_size, radius, fixed_kernel_points))

    def initialize_kernel_points(self, kernel_size, radius, fixed):
        # For simplicity we distribute points uniformly (using random normals) on the sphere
        # If fixed == 'center', the first kernel point is set to the origin.
        kp = torch.randn(kernel_size, 3)
        kp = kp / kp.norm(dim=1, keepdim=True) * radius
        if fixed == 'center':
            kp[0] = torch.zeros(3)
        return kp

    def index_points(self, points, idx):
        """
        points: (B, M, C), idx: (B, N, k) -> output: (B, N, k, C)
        """
        B = points.shape[0]
        # create batch indices
        batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
        return points[batch_indices, idx, :]

    def forward(self, query_points, support_points, neighbor_idx, features):
        """
        See class docstring.
        """
        B, N, _ = query_points.shape
        k = neighbor_idx.shape[-1]  # number of neighbors
        # Gather neighbor features and positions (B, N, k, C) and (B, N, k, 3)
        neighbor_feats = self.index_points(features, neighbor_idx)  # (B, N, k, in_channels)
        neighbor_coords = self.index_points(support_points, neighbor_idx)  # (B, N, k, 3)
        # Compute relative coordinates (B, N, k, 3)
        query_expanded = query_points.unsqueeze(2)  # (B, N, 1, 3)
        rel_coords = neighbor_coords - query_expanded  # (B, N, k, 3)
        # For each kernel point, compute the distance from the kernel point to each neighbor’s relative coordinate
        # kernel_points: (kernel_size, 3) -> (1, 1, 1, kernel_size, 3)
        kp = self.kernel_points.view(1, 1, 1, self.kernel_size, 3)
        # Expand relative coordinates: (B, N, k, 1, 3)
        rel_coords_exp = rel_coords.unsqueeze(3)
        # Compute L2 distance: (B, N, k, kernel_size)
        dists = torch.norm(rel_coords_exp - kp, dim=-1)
        # Compute influence weights for each neighbor and each kernel point
        if self.influence == 'linear':
            weight_influences = torch.clamp(1 - dists / self.radius, min=0)
        elif self.influence == 'gaussian':
            sigma = self.radius * 0.3
            weight_influences = torch.exp(- (dists ** 2) / (2 * sigma ** 2))
        else:  # constant
            weight_influences = torch.ones_like(dists)
        # weight_influences: (B, N, k, kernel_size)
        # Multiply the influences with neighbor features.
        # First, expand neighbor_feats to (B, N, k, 1, in_channels)
        neighbor_feats_exp = neighbor_feats.unsqueeze(3)
        # Multiply: (B, N, k, kernel_size, in_channels)
        weighted_feats = weight_influences.unsqueeze(-1) * neighbor_feats_exp
        # Sum over neighbors: (B, N, kernel_size, in_channels)
        aggregated = weighted_feats.sum(dim=2)
        # Now apply the learned weights (kernel aggregation)
        # aggregated: (B, N, kernel_size, in_channels)
        # weights: (kernel_size, in_channels, out_channels)
        # Use einsum: output[b, n, o] = sum_{i, j} aggregated[b, n, i, j] * weights[i, j, o]
        # output = torch.einsum('bnki,iko->bno', aggregated, self.weights)
        output = torch.einsum('bnki,kio->bno', aggregated, self.weights)

        return output

# class KPConvEncoder(nn.Module):
#     """
#     KPConv–based encoder to replace your current PointNet (or PointNet++ style) encoder.
#     This module takes as input a point cloud (B, N, 3) and returns a dictionary with keys:
#        'mu_1d' and 'sigma_1d' (each B, zdim)
#     """
#     def __init__(self, zdim, input_dim=3, num_neighbors=16,
#                  kconv_channels=[32, 64, 128],
#                  kernel_size=15, radius=0.1):
#         """
#         Args:
#           zdim: dimension of the latent code.
#           input_dim: dimensionality of the input points (typically 3).
#           num_neighbors: number of neighbors to group (for KPConv).
#           kconv_channels: list of output channel sizes for successive KPConv layers.
#           kernel_size: number of kernel points in each KPConv layer.
#           radius: influence radius for the KPConv layers.
#         """
#         super(KPConvEncoder, self).__init__()
#         self.zdim = zdim
#         self.input_dim = input_dim
#         self.num_neighbors = num_neighbors

#         # (Optional) initial feature embedding from coordinates
#         self.initial_mlp = nn.Sequential(
#             nn.Linear(input_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, 16)
#         )

#         layers = []
#         in_channels = 16  # after the initial MLP
#         for out_channels in kconv_channels:
#             layers.append(KPConv(in_channels, out_channels, kernel_size, radius))
#             # You can add a non-linearity here; we do it in the forward pass.
#             in_channels = out_channels
#         self.kconv_layers = nn.ModuleList(layers)
#         self.fc = nn.Linear(in_channels, zdim * 2)

#     def forward(self, x):
#         """
#         Args:
#           x: (B, N, 3)
#         Returns:
#           A dictionary with keys 'mu_1d' and 'sigma_1d' (each of shape (B, zdim)).
#         """
#         B, N, _ = x.shape
#         # Get an initial feature embedding
#         feats = self.initial_mlp(x)  # (B, N, 16)

#         # Compute neighbor indices (using simple kNN on the raw coordinates)
#         neighbor_idx = get_neighbors(x, self.num_neighbors)  # (B, N, num_neighbors)

#         # Loop over KPConv layers
#         for layer in self.kconv_layers:
#             conv_out = layer(x, x, neighbor_idx, feats)  # (B, N, out_channels)
#             conv_out = F.relu(conv_out)
#             feats = conv_out  # update features

#         # Global feature aggregation (max pooling)
#         global_feat, _ = torch.max(feats, dim=1)  # (B, C)
#         out = self.fc(global_feat)  # (B, 2*zdim)
#         mu = out[:, :self.zdim]
#         sigma = out[:, self.zdim:]
#         return {'mu_1d': mu, 'sigma_1d': sigma}

# In kpconv_encoder.py
class KPConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius, activation=nn.ReLU()):
        super(KPConvBlock, self).__init__()
        self.kpconv = KPConv(in_channels, out_channels, kernel_size, radius)
        self.activation = activation
        
    def forward(self, query_points, support_points, neighbor_idx, feats):
        out = self.kpconv(query_points, support_points, neighbor_idx, feats)
        if self.activation is not None:
            out = self.activation(out)
        return out

class KPConvEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, num_neighbors=16,
                 kconv_channels=[32, 64, 128],
                 kernel_size=15, radius=0.1, **kwargs):
        super(KPConvEncoder, self).__init__()
        self.zdim = zdim
        self.input_dim = input_dim
        self.num_neighbors = num_neighbors

        # Optional: initial embedding of coordinates
        self.initial_mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        layers = []
        in_channels = 16  # output from initial_mlp
        for out_channels in kconv_channels:
            layers.append(KPConvBlock(in_channels, out_channels, kernel_size, radius, activation=nn.ReLU()))
            in_channels = out_channels
        self.kconv_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(in_channels, zdim * 2)

    def forward(self, x):
        B, N, _ = x.shape
        feats = self.initial_mlp(x)  # (B, N, 16)
        neighbor_idx = get_neighbors(x, self.num_neighbors)  # (B, N, num_neighbors)
        # Apply each KPConv block sequentially.
        for block in self.kconv_layers:
            feats = block(x, x, neighbor_idx, feats)
        # Global max pooling over points
        global_feat, _ = torch.max(feats, dim=1)
        out = self.fc(global_feat)
        mu = out[:, :self.zdim]
        sigma = out[:, self.zdim:]
        return {'mu_1d': mu, 'sigma_1d': sigma}
