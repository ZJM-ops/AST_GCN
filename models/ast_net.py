#ast_net.py
import timm
import torch
import torch.nn as nn


def build_hippocampal_adjacency():
    """
    构8通道海马体LFP的解剖学邻接矩阵
    
    通道映射:
    0: CA1(L)  1: CA1(R)  2: CA3(L)  3: CA3(R)
    4: DG(L)   5: DG(R)   6: Ref(L)  7: Ref(R)
    
    连接强度设计:
    1. 海马回路: DG → CA3 → CA1 (强)
    2. 左右对称: 同一区域左右强连接
    3. 参考电极: 全局基准，弱连接所有通道
    """
    adj = torch.zeros(8, 8)
    
    # 1. 同侧海马回路（解剖学强连接）
    # DG(L) → CA3(L) → CA1(L)
    adj[4, 2] = 1.0   # DG(L) → CA3(L) (mossy fibers)
    adj[2, 0] = 1.0   # CA3(L) → CA1(L) (Schaffer collateral)
    
    # DG(R) → CA3(R) → CA1(R)
    adj[5, 3] = 1.0   # DG(R) → CA3(R)
    adj[3, 1] = 1.0   # CA3(R) → CA1(R)
    
    # 2. 左右对称连接（commissural connections）
    # CA3间有强连合纤维，CA1和DG间有弱连接
    adj[2, 3] = 0.8   # CA3(L) ↔ CA3(R) 强
    adj[3, 2] = 0.8
    adj[0, 1] = 0.6   # CA1(L) ↔ CA1(R) 中
    adj[1, 0] = 0.6
    adj[4, 5] = 0.5   # DG(L) ↔ DG(R) 弱
    adj[5, 4] = 0.5
    
    # 3. 反馈连接（可学习，初始弱）
    adj[0, 2] = 0.3
    adj[1, 3] = 0.3
    adj[2, 4] = 0.2
    adj[3, 5] = 0.2
    
    # 4. 参考电极（全局基准）
    adj[6, :] = 0.1   # Ref(L) → all
    adj[:, 6] = 0.1   # all → Ref(L)
    adj[7, :] = 0.1   # Ref(R) → all
    adj[:, 7] = 0.1   # all → Ref(R)
    adj[6, 7] = 0.5   # Ref(L) ↔ Ref(R)
    adj[7, 6] = 0.5
    
    # 5. 自连接
    adj = adj + torch.eye(8) * 0.5
    
    return adj


def build_chbmit_scalp_adjacency():
    """
    构建CHB-MIT 18通道双极导联的头皮拓扑邻接矩阵
    
    通道顺序 (标准10-20纵向双极导联, double banana):
      左外侧链:    FP1-F7, F7-T7, T7-P7, P7-O1       (0-3)
      左旁正中链:  FP1-F3, F3-C3, C3-P3, P3-O1       (4-7)
      右旁正中链:  FP2-F4, F4-C4, C4-P4, P4-O2       (8-11)
      右外侧链:    FP2-F8, F8-T8, T8-P8, P8-O2       (12-15)
      中线链:      FZ-CZ, CZ-PZ                       (16-17)
    
    连接强度设计 (与海马体邻接同哲学):
      1. 同链相邻 (共享电极): 1.0  — 双极导联标志性连接
      2. 跨链共享电极: 0.7        — 额极/枕极交叉
      3. 左右对称: 0.6            — 跨半球
      4. 旁正中↔中线同级: 0.4
      5. 自连接: 0.5
    """
    channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',     # 左外侧
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',     # 左旁正中
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',     # 右旁正中
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',     # 右外侧
        'FZ-CZ', 'CZ-PZ'                          # 中线
    ]
    n = len(channels)
    electrodes = [ch.split('-') for ch in channels]
    adj = torch.zeros(n, n)

    # 5条前-后走向的链
    chains = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17]]

    # 1. 同链相邻 (共享电极) - 最强
    for chain in chains:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            adj[a, b] = 1.0
            adj[b, a] = 1.0

    # 2. 跨链共享电极 (额极/枕极交叉)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                continue
            if set(electrodes[i]) & set(electrodes[j]):
                adj[i, j] = 0.7
                adj[j, i] = 0.7

    # 3. 左右对称 (同水平不同半球对应通道)
    sym_pairs = [(0, 12), (1, 13), (2, 14), (3, 15),   # 左外侧 ↔ 右外侧
                 (4, 8), (5, 9), (6, 10), (7, 11)]     # 左旁正中 ↔ 右旁正中
    for a, b in sym_pairs:
        if adj[a, b] == 0:
            adj[a, b] = 0.6
            adj[b, a] = 0.6

    # 4. 旁正中 ↔ 中线 (同矢状位水平)
    midline_pairs = [(5, 16), (6, 17),   # 左旁正中 ↔ 中线
                     (9, 16), (10, 17)]  # 右旁正中 ↔ 中线
    for a, b in midline_pairs:
        if adj[a, b] == 0:
            adj[a, b] = 0.4
            adj[b, a] = 0.4

    # 5. 自连接
    adj = adj + torch.eye(n) * 0.5
    return adj


class MultiScaleResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(MultiScaleResNeXtBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.scale_width = width // 4
        self.last_width = width - (self.scale_width * 3)

        self.conv_branch1 = nn.Conv2d(self.scale_width, self.scale_width,
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_branch2 = nn.Conv2d(self.scale_width, self.scale_width,
                                      kernel_size=3, stride=stride, padding=1,
                                      groups=self.scale_width // 2 if self.scale_width > 2 else 1, bias=False)
        self.conv_branch3 = nn.Conv2d(self.scale_width, self.scale_width,
                                      kernel_size=(3, 11), stride=stride, padding=(1, 5),
                                      groups=self.scale_width // 2 if self.scale_width > 2 else 1, bias=False)
        self.conv_branch4 = nn.Conv2d(self.last_width, self.last_width,
                                      kernel_size=(7, 1), stride=stride, padding=(3, 0),
                                      groups=self.last_width // 2 if self.last_width > 2 else 1, bias=False)

        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))

        x1, x2, x3, x4 = torch.split(out, [self.scale_width, self.scale_width, self.scale_width, self.last_width], dim=1)
        out = torch.cat([
            self.conv_branch1(x1),
            self.conv_branch2(x2),
            self.conv_branch3(x3),
            self.conv_branch4(x4)
        ], dim=1)

        out = self.relu(self.bn2(out))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


def _create_checkpoint_wrapper(forward_fn):
    def wrapper(x):
        return torch.utils.checkpoint.checkpoint(forward_fn, x, use_reentrant=False)
    return wrapper

def apply_multiscale_block(model, gradient_checkpointing=False):
    from timm.models.resnet import Bottleneck as TimmBottleneck
    from timm.models.resnet import BasicBlock as TimmBasicBlock
    for name, module in model.named_modules():
        if name.startswith('layer') and len(list(module.children())) > 0:
            for i, child in enumerate(module.children()):
                if isinstance(child, TimmBottleneck):
                    inplanes = child.conv1.in_channels
                    expansion = child.expansion
                    original_planes = child.conv3.out_channels // expansion
                    stride = child.conv2.stride[0] if hasattr(child.conv2, 'stride') else 1
                    groups = child.conv2.groups if hasattr(child.conv2, 'groups') else 1
                    new_block = MultiScaleResNeXtBlock(
                        inplanes=inplanes, planes=original_planes,
                        stride=stride, downsample=child.downsample,
                        groups=groups, base_width=32,
                        norm_layer=type(child.bn1)
                    )
                    if gradient_checkpointing:
                        new_block.forward = _create_checkpoint_wrapper(new_block.forward)
                    module[i] = new_block
                elif isinstance(child, TimmBasicBlock) and gradient_checkpointing:
                    child.forward = _create_checkpoint_wrapper(child.forward)
    return model


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return torch.bmm(adj, self.linear(x))


class ResNeXt_GCN_Bridge(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, feature_dim=2048, backbone_name='resnext50_32x4d', gradient_checkpointing=False, use_anatomical_prior=True):
        super(ResNeXt_GCN_Bridge, self).__init__()
        self.num_nodes = num_nodes
        self.use_anatomical_prior = use_anatomical_prior

        base_model = timm.create_model(backbone_name, num_classes=0)
        self.backbone = apply_multiscale_block(base_model, gradient_checkpointing)
        self.gradient_checkpointing = gradient_checkpointing
        
        self.feature_dim = base_model.num_features if hasattr(base_model, 'num_features') else feature_dim

        self.attn_query = nn.Linear(self.feature_dim, 128)
        self.attn_key = nn.Linear(self.feature_dim, 128)

        if use_anatomical_prior:
            if num_nodes == 8:
                prior_adj = build_hippocampal_adjacency()
            elif num_nodes == 18:
                prior_adj = build_chbmit_scalp_adjacency()
            else:
                prior_adj = None
            if prior_adj is not None:
                self.register_buffer('anatomical_adj', prior_adj)
                self.prior_weight = nn.Parameter(torch.tensor(0.3))
                print(f"🧠 [AST-Net_GCN] 启用解剖学位置先验 (num_nodes={num_nodes}, 邻接shape={tuple(prior_adj.shape)})")
            else:
                self.anatomical_adj = None
        else:
            self.anatomical_adj = None

        self.gcn1 = GCNLayer(self.feature_dim, 512)
        self.gcn2 = GCNLayer(512, 128)
        self.relu = nn.ReLU(inplace=True)

        self.channel_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size * self.num_nodes, 3, 224, 224)
        
        if self.gradient_checkpointing:
            def backbone_forward(x_f):
                return self.backbone(x_f)
            backbone_out = torch.utils.checkpoint.checkpoint(backbone_forward, x_flat, use_reentrant=False)
        else:
            backbone_out = self.backbone(x_flat)
        
        node_features = backbone_out.view(batch_size, self.num_nodes, -1)

        q = self.attn_query(node_features)
        k = self.attn_key(node_features).transpose(1, 2)
        adj = torch.softmax(torch.bmm(q, k), dim=-1)
        
        if self.anatomical_adj is not None:
            prior_adj = self.anatomical_adj.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
            adj = adj + torch.sigmoid(self.prior_weight) * prior_adj
        
        iden = torch.eye(self.num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + iden
        
        degree = adj.sum(dim=-1, keepdim=True)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        adj_norm = degree_inv_sqrt * adj * degree_inv_sqrt.transpose(1, 2)

        g_out = self.relu(self.gcn1(node_features, adj_norm))
        g_out = self.relu(self.gcn2(g_out, adj_norm))
        
        g_out_attn, _ = self.channel_attention(g_out, g_out, g_out)
        g_out = g_out + g_out_attn
        
        return self.fc(g_out.mean(dim=1))