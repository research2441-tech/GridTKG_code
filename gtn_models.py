"""
Graph temporal network (GTN) and the comparison models used by the analytics
layer.  All models map a window X of shape (B, W, N, F) to one logit per node
(B, N).  The graph is the materialised adjacentTo relation of the knowledge
graph (branches that share a bus), normalised as D^-1/2 (A + I) D^-1/2.
"""
import numpy as np
import torch
import torch.nn as nn


def norm_adj(src, dst, n):
    """Symmetric normalised adjacency with self-loops as a torch sparse tensor."""
    s = np.r_[src, np.arange(n)]
    d = np.r_[dst, np.arange(n)]
    deg = np.bincount(s, minlength=n).astype(np.float32)
    w = 1.0 / np.sqrt(deg[s] * deg[d])
    idx = torch.tensor(np.vstack([s, d]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(w, dtype=torch.float32), (n, n)).coalesce()


def spmm_batch(A, H):
    """A: (N,N) sparse; H: (..., N, C) -> (..., N, C)."""
    shp = H.shape
    N, Cc = shp[-2], shp[-1]
    Hm = H.reshape(-1, N, Cc).permute(1, 0, 2).reshape(N, -1)
    out = torch.sparse.mm(A, Hm)
    return out.reshape(N, -1, Cc).permute(1, 0, 2).reshape(shp)


class GCNLayer(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(cin, cout)
        self.skip = nn.Linear(cin, cout, bias=False)

    def forward(self, A, H):
        return torch.relu(spmm_batch(A, self.lin(H)) + self.skip(H))


class MLP(nn.Module):
    """No graph, no time: uses the last step of the window only."""
    def __init__(self, f, h=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, A=None):
        return self.net(X[:, -1]).squeeze(-1)


class GRUNode(nn.Module):
    """Temporal model per node, no graph."""
    def __init__(self, f, h=48):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, A=None):
        B, W, N, F = X.shape
        z = torch.relu(self.inp(X)).permute(0, 2, 1, 3).reshape(B * N, W, -1)
        _, hN = self.gru(z)
        return self.out(hN[-1]).reshape(B, N)


class GCN(nn.Module):
    """Static graph model ('traditional GNN'): last step of the window only."""
    def __init__(self, f, h=48, layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.layers = nn.ModuleList([GCNLayer(h, h) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, A):
        x = torch.relu(self.inp(X[:, -1]))
        h = x
        for l in self.layers:
            h = l(A, h)
        return self.out(torch.cat([x, h], -1)).squeeze(-1)


class GTN(nn.Module):
    """Graph temporal network: per-step graph convolution followed by a GRU
    over the window; the readout combines the node's own temporal state with
    the graph-aggregated temporal state."""
    def __init__(self, f, h=48, layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.layers = nn.ModuleList([GCNLayer(h, h) for _ in range(layers)])
        self.gru = nn.GRU(2 * h, h, batch_first=True)
        self.post = GCNLayer(h, h)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, A):
        B, W, N, F = X.shape
        x = torch.relu(self.inp(X))                     # (B,W,N,h)
        h = x
        for l in self.layers:
            h = l(A, h)
        z = torch.cat([x, h], -1).permute(0, 2, 1, 3).reshape(B * N, W, -1)
        _, hN = self.gru(z)
        s = hN[-1].reshape(B, N, -1)
        g = self.post(A, s)
        return self.out(torch.cat([s, g], -1)).squeeze(-1)


MODELS = dict(MLP=MLP, GRU=GRUNode, GCN=GCN, GTN=GTN)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
