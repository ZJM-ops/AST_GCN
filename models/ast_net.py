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
        
        # SE模块：跨尺度通道注意力融合
        self.se_reduction = width // 16
        self.se_fc1 = nn.Conv2d(width, self.se_reduction, kernel_size=1, bias=False)
        self.se_fc2 = nn.Conv2d(self.se_reduction, width, kernel_size=1, bias=False)
        self.se_sigmoid = nn.Sigmoid()
        
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

        # SE模块：自适应跨尺度融合
        se_out = torch.mean(out, dim=[2, 3], keepdim=True)  # 全局平均池化
        se_out = self.se_sigmoid(self.se_fc2(self.relu(self.se_fc1(se_out))))
        out = out * se_out  # 通道加权
        
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
    def __init__(self, in_features, out_features, dropout=0.3):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        # 残差连接
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x, adj):
        out = torch.bmm(adj, self.linear(x))
        out = self.ln(out)
        # 残差连接
        if self.residual is not None:
            out = out + self.residual(x)
        else:
            out = out + x
        return self.dropout(out)


class ResNeXt_GCN_Bridge(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, feature_dim=2048, backbone_name='resnext50_32x4d', gradient_checkpointing=False, use_anatomical_prior=True):
        super(ResNeXt_GCN_Bridge, self).__init__()
        self.num_nodes = num_nodes
        self.use_anatomical_prior = use_anatomical_prior

        base_model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.backbone = apply_multiscale_block(base_model, gradient_checkpointing)
        self.gradient_checkpointing = gradient_checkpointing
        
        self.feature_dim = base_model.num_features if hasattr(base_model, 'num_features') else feature_dim

        self.num_heads = 4
        self.attn_dim = 64
        self.attn_query = nn.Linear(self.feature_dim, self.num_heads * self.attn_dim)
        self.attn_key = nn.Linear(self.feature_dim, self.num_heads * self.attn_dim)

        if use_anatomical_prior and num_nodes == 8:
            self.register_buffer('anatomical_adj', build_hippocampal_adjacency())
            self.prior_weight = nn.Parameter(torch.tensor(0.3))
        else:
            self.anatomical_adj = None

        self.gcn1 = GCNLayer(self.feature_dim, 512, dropout=0.2)
        self.gcn2 = GCNLayer(512, 128, dropout=0.2)
        self.relu = nn.ReLU(inplace=True)

        self.channel_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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

        q = self.attn_query(node_features).view(batch_size, self.num_nodes, self.num_heads, self.attn_dim)
        k = self.attn_key(node_features).view(batch_size, self.num_nodes, self.num_heads, self.attn_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        adj = torch.softmax(attn_scores, dim=-1)
        adj = adj.mean(dim=1)
        
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