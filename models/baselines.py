import timm
import torch
import torch.nn as nn

from models.ast_net import apply_multiscale_block


class BaselineMultiScaleResNeXt(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, backbone_name='resnext50_32x4d', gradient_checkpointing=False):
        super(BaselineMultiScaleResNeXt, self).__init__()
        self.num_nodes = num_nodes
        self.backbone = timm.create_model(backbone_name, num_classes=0)
        self.backbone = apply_multiscale_block(self.backbone, gradient_checkpointing)
        feat_dim = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * num_nodes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.num_nodes, 3, 224, 224)
        feats = self.backbone(x).view(B, -1)
        return self.fc(feats)


class BaselineResNeXt(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, backbone_name='resnext50_32x4d', gradient_checkpointing=False):
        super(BaselineResNeXt, self).__init__()
        self.num_nodes = num_nodes
        self.backbone = timm.create_model(backbone_name, num_classes=0)
        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing()
        feat_dim = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * num_nodes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.num_nodes, 3, 224, 224)
        feats = self.backbone(x).view(B, -1)
        return self.fc(feats)


class BaselineResNet(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, backbone_name='resnet18', gradient_checkpointing=False):
        super(BaselineResNet, self).__init__()
        self.num_nodes = num_nodes
        self.backbone = timm.create_model(backbone_name, num_classes=0)
        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing()
        feat_dim = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * num_nodes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.num_nodes, 3, 224, 224)
        feats = self.backbone(x).view(B, -1)
        return self.fc(feats)


class BaselineEfficientNet(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, backbone_name='efficientnet_b0', gradient_checkpointing=False):
        super(BaselineEfficientNet, self).__init__()
        self.num_nodes = num_nodes
        self.backbone = timm.create_model(backbone_name, num_classes=0)
        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing()
        feat_dim = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * num_nodes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.num_nodes, 3, 224, 224)
        feats = self.backbone(x).view(B, -1)
        return self.fc(feats)


class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_classes=3, num_nodes=18, backbone_name='resnet18', gradient_checkpointing=False):
        super(BaselineCNNLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.cnn = timm.create_model(backbone_name, num_classes=0)
        if gradient_checkpointing:
            self.cnn.set_grad_checkpointing()
        feat_dim = self.cnn.num_features
        self.lstm = nn.LSTM(feat_dim, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.num_nodes, 3, 224, 224)
        feats = self.cnn(x).view(B, self.num_nodes, -1)
        lstm_out, _ = self.lstm(feats)
        out = lstm_out[:, -1, :]
        return self.fc(out)