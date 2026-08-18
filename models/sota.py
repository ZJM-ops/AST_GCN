import numpy as np
import torch
import torch.nn as nn

try:
    from braindecode.models import EEGConformer as BrainEEGConformer
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False


class EEGNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=8, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()
        self.num_channels = num_channels

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F2, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


class TGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, dropout=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            bias=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv, kvw -> nctw', x, A)
        return self.drop(x.contiguous())


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5, residual=True):
        super().__init__()
        self.gcn = TGCN(in_channels, out_channels, kernel_size, dropout=dropout)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class ST_GCN_EEG(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, num_nodes=18, dropout=0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.kernel_size = 1

        A = torch.ones(self.kernel_size, num_nodes, num_nodes)
        self.register_buffer('A', A)
        self.PA = nn.Parameter(torch.from_numpy(np.ones((self.kernel_size, num_nodes, num_nodes))).float(), requires_grad=True)

        self.data_bn = nn.BatchNorm2d(in_channels)
        self.st_gcn_networks = nn.ModuleList((
            STGCN_Block(in_channels, 32, self.kernel_size, stride=1, dropout=dropout, residual=False),
            STGCN_Block(32, 32, self.kernel_size, stride=2, dropout=dropout),
            STGCN_Block(32, 64, self.kernel_size, stride=2, dropout=dropout),
            STGCN_Block(64, 64, self.kernel_size, stride=2, dropout=dropout)
        ))
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks
        ])
        self.fcn = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.data_bn(x)
        A = self.A * self.PA

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, A * importance)

        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = self.fcn(x)
        x = torch.flatten(x, 1)
        return x


class ResConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.drop(self.gelu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.gelu(out + residual)


class EEGConformer1DEncoder(nn.Module):
    def __init__(self, num_nodes=18, embed_dim=256, time_points=1000):
        super().__init__()
        self.num_nodes = num_nodes

        self.time_conv = nn.Sequential(
            ResConv1d(1, 32, kernel=25, stride=2),
            ResConv1d(32, 64, kernel=15, stride=2),
            nn.AdaptiveAvgPool1d(64)
        )

        self.projection = nn.Linear(64, embed_dim)
        self.drop_in = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, x):
        batch_size = x.size(0)
        time_len = x.size(2)
        x = x.view(batch_size * self.num_nodes, 1, time_len)
        cnn_out = self.time_conv(x)
        tokens = self.drop_in(self.projection(cnn_out))
        tx_out = self.transformer(tokens)
        node_features = tx_out.mean(dim=1)
        node_features = node_features.view(batch_size, self.num_nodes, -1)
        return node_features


class GCNLayer1D(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, adj):
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj / degree
        support = self.linear(x)
        graph_agg = torch.bmm(adj_norm, support)
        out = self.drop(self.norm(graph_agg + support))
        return out


class EEGConformer1D_GCN_Model(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, embed_dim=256, time_points=1000):
        super().__init__()
        self.num_nodes = num_nodes
        self.conformer_1d = EEGConformer1DEncoder(num_nodes=num_nodes, embed_dim=embed_dim, time_points=time_points)

        self.attn_query = nn.Linear(embed_dim, 64)
        self.attn_key = nn.Linear(embed_dim, 64)
        self.adj_drop = nn.Dropout(0.15)

        self.gcn1 = GCNLayer1D(embed_dim, 128)
        self.proj1 = nn.Linear(embed_dim, 128)
        self.gcn2 = GCNLayer1D(128, 64)
        self.proj2 = nn.Linear(128, 64)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(num_nodes * 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        node_features = self.conformer_1d(x)

        q = self.attn_query(node_features)
        k = self.attn_key(node_features).transpose(1, 2)
        adj = torch.bmm(q, k)
        adj = torch.softmax(adj, dim=-1)
        adj = self.adj_drop(adj)
        iden = torch.eye(self.num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj = adj + iden

        g1_out = self.gcn1(node_features, adj)
        residual1 = self.proj1(node_features)
        g1 = self.relu(g1_out + residual1)

        g2_out = self.gcn2(g1, adj)
        residual2 = self.proj2(g1)
        g2 = self.relu(g2_out + residual2)

        flat = g2.view(batch_size, -1)
        out = self.fc(flat)
        return out


class BraindecodeEEGConformer(nn.Module):
    def __init__(self, num_classes=3, num_channels=8, n_times=2000,
                 n_filters_time=40, filter_time_length=25,
                 pool_time_length=75, pool_time_stride=15,
                 drop_prob=0.5, num_layers=6, num_heads=10):
        super().__init__()
        if not BRAINDECODE_AVAILABLE:
            raise ImportError("braindecode 库未安装，请运行: pip install braindecode")
        self.model = BrainEEGConformer(
            n_chans=num_channels,
            n_outputs=num_classes,
            n_times=n_times,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def forward(self, x):
        return self.model(x)