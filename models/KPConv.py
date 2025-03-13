# KPConv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def square_distance(src, dst):
    # src: [B, N, C], dst: [B, M, C]
    B, N, C = src.shape
    _, M, _ = dst.shape
    # Compute pairwise squared distance: (x-y)^2 = x^2 + y^2 - 2xy
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(2, 1)
    return dist

def ball_query(radius, nsample, xyz, new_xyz):
    # xyz: [B, N, 3], new_xyz: [B, S, 3]
    # Returns indices of neighbors: [B, S, nsample]
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    # Compute squared distances between each query point and all points
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    # Get the nsample nearest neighbors (by sorting)
    group_idx = sqrdists.argsort()[:, :, :nsample]  # [B, S, nsample]
    # Optionally, you could mask neighbors with distance > radius^2.
    # Here we simply replace them with the first neighbor index.
    grouped_dists = torch.gather(sqrdists, 2, group_idx)
    mask = grouped_dists > (radius ** 2)
    group_idx[mask] = group_idx[:, :, 0:1].expand_as(group_idx)[mask]
    return group_idx

class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernel_points=15, radius=1.0, 
                 kp_influence='linear', fixed_kernel_points='center'):
        """
        Args:
          in_channels: input feature dimension.
          out_channels: output feature dimension.
          num_kernel_points: number of kernel points.
          radius: neighborhood radius.
          kp_influence: influence function: 'linear', 'gaussian' or 'constant'.
          fixed_kernel_points: if 'center', the first kernel point is fixed at 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points
        self.radius = radius
        self.kp_influence = kp_influence
        self.fixed_kernel_points = fixed_kernel_points
        # Learnable weights: [K, in_channels, out_channels]
        # self.weights = nn.Parameter(torch.randn(num_kernel_points, in_channels, out_channels))
        # self.weights = nn.Parameter(torch.empty(num_kernel_points, in_channels, out_channels))
        self.weights = nn.Parameter(torch.randn(num_kernel_points, in_channels, out_channels) * 0.005)

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Initialize kernel point positions (fixed)
        self.register_buffer('kernel_points', self.init_kernel_points(num_kernel_points, radius, fixed_kernel_points))
    
    def init_kernel_points(self, num_kernel_points, radius, fixed):
        # Uniformly sample points on the unit sphere then scale by radius.
        kp = torch.randn(num_kernel_points, 3)
        kp = F.normalize(kp, p=2, dim=1) * radius
        if fixed == 'center':
            kp[0] = torch.zeros(3)  # force first kernel point to be at the center
        return kp

    # def forward(self, query_xyz, features):
    #     """
    #     Args:
    #       query_xyz: [B, S, 3] coordinates of query points (and centers for grouping).
    #       features: [B, N, in_channels] input features (usually N equals S if no subsampling).
    #     Returns:
    #       (query_xyz, out_features) where out_features is [B, S, out_channels].
    #     """
    #     B, S, _ = query_xyz.shape
    #     # For simplicity, we assume the input points are the query points.
    #     # In practice, one may use a separate function to perform grouping.
    #     group_idx = ball_query(self.radius, nsample=32, xyz=query_xyz, new_xyz=query_xyz)  # [B, S, nsample]
    #     # Gather neighbor features [B, S, nsample, in_channels]
    #     # (Assume features and coordinates align; if not, adapt accordingly)
    #     B, N, C = features.shape
    #     # Use torch.gather along the N dimension:
    #     idx_expanded = group_idx.unsqueeze(-1).expand(B, S, group_idx.shape[-1], C)
    #     grouped_features = torch.gather(features, 1, idx_expanded)
    #     # Compute relative positions: [B, S, nsample, 3]
    #     grouped_xyz = torch.gather(query_xyz, 1, group_idx.unsqueeze(-1).expand(B, S, group_idx.shape[-1], 3))
    #     relative_xyz = grouped_xyz - query_xyz.unsqueeze(2)
    #     # Compute distances from each neighbor to each kernel point.
    #     # Expand dimensions for broadcasting: 
    #     # relative_xyz: [B, S, nsample, 1, 3], kernel_points: [1, 1, 1, K, 3]
    #     diff = relative_xyz.unsqueeze(3) - self.kernel_points.view(1, 1, 1, self.num_kernel_points, 3)
    #     sq_dist = torch.sum(diff ** 2, dim=-1)  # [B, S, nsample, K]
    #     if self.kp_influence == 'linear':
    #         influence = torch.clamp(1 - torch.sqrt(sq_dist) / self.radius, min=0)
    #     elif self.kp_influence == 'gaussian':
    #         sigma = self.radius * 0.3
    #         influence = torch.exp(-sq_dist / (2 * sigma * sigma))
    #     else:  # constant
    #         influence = torch.ones_like(sq_dist)
    #     # Now, weight the neighbor features by the influence.
    #     # Expand grouped features: [B, S, nsample, 1, in_channels]
    #     weighted_features = grouped_features.unsqueeze(3) * influence.unsqueeze(-1)
    #     # Sum over neighbors: [B, S, K, in_channels]
    #     agg_features = weighted_features.sum(dim=2)
    #     # Apply the kernel weights:
    #     # weights: [K, in_channels, out_channels]
    #     # Use einsum to get: [B, S, K, out_channels]
    #     out = torch.einsum('bski,kio->bsko', agg_features, self.weights)
    #     # Sum over the kernel points:
    #     out = out.sum(dim=2)  # [B, S, out_channels]
    #     return query_xyz, out
    def forward(self, query_xyz, features):
        # query_xyz: [B, S, 3]
        # features: [B, N, in_channels] (assuming N = S if no subsampling)
        B, S, _ = query_xyz.shape
        nsample = 32  # number of neighbors to sample

        # Get neighbor indices using ball query (assume using query_xyz for both new_xyz and xyz)
        group_idx = ball_query(self.radius, nsample, xyz=query_xyz, new_xyz=query_xyz)  # [B, S, nsample]

        # Expand features from [B, N, C] to [B, S, N, C]
        features_expanded = features.unsqueeze(1).expand(B, S, -1, -1)  # [B, S, N, C]
        # Expand index tensor to [B, S, nsample, C] to match features_expanded's dimensions along dim=2
        idx_expanded = group_idx.unsqueeze(-1).expand(B, S, nsample, features.size(-1))  # [B, S, nsample, C]
        # Gather neighbor features -> [B, S, nsample, in_channels]
        grouped_features = torch.gather(features_expanded, 2, idx_expanded)
        
        # For relative positions, gather neighbor coordinates in a similar way:
        query_xyz_expanded = query_xyz.unsqueeze(2).expand(B, S, nsample, 3)  # [B, S, nsample, 3]
        # Here, since the point coordinates are the same as query_xyz, we can also use group_idx similarly:
        idx_xyz = group_idx.unsqueeze(-1).expand(B, S, nsample, 3)
        grouped_xyz = torch.gather(query_xyz.unsqueeze(1).expand(B, S, -1, 3), 2, idx_xyz)  # [B, S, nsample, 3]
        
        # Compute relative positions: [B, S, nsample, 3]
        relative_xyz = grouped_xyz - query_xyz_expanded
        
        # Compute differences to each kernel point:
        # relative_xyz: [B, S, nsample, 1, 3]
        # kernel_points: [1, 1, 1, K, 3]
        diff = relative_xyz.unsqueeze(3) - self.kernel_points.view(1, 1, 1, self.num_kernel_points, 3)
        # Square distances: [B, S, nsample, K]
        # sq_dist = torch.sum(diff ** 2, dim=-1)
        sq_dist = torch.clamp(torch.sum(diff ** 2, dim=-1), min=0)

        
        # Compute influence weights according to the influence function
        if self.kp_influence == 'linear':
            influence = torch.clamp(1 - torch.sqrt(sq_dist + 1e-8) / (self.radius + 1e-8), min=0)
            # influence = torch.clamp(1 - torch.sqrt(sq_dist) / self.radius, min=0)
        elif self.kp_influence == 'gaussian':
            sigma = self.radius * 0.3
            influence = torch.exp(-sq_dist / (2 * sigma * sigma))
        else:  # constant
            influence = torch.ones_like(sq_dist)
        
        # Weight neighbor features with influence: expand grouped_features: [B, S, nsample, 1, C]
        weighted_features = grouped_features.unsqueeze(3) * influence.unsqueeze(-1)
        # Sum over the neighbors: [B, S, K, in_channels]
        agg_features = weighted_features.sum(dim=2)
        # Apply kernel weights using Einstein summation: [B, S, K, out_channels]
        out = torch.einsum('bski,kio->bsko', agg_features, self.weights)
        # Sum over the kernel points: [B, S, out_channels]
        out = out.sum(dim=2)
        
        return query_xyz, out
