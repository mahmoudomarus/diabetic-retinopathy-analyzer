import torch
import torch.nn as nn
from transformers import ViTModel
from efficientnet_pytorch import EfficientNet

class EnhancedRetinalAnalyzer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        vit_dim = self.vit.config.hidden_size
        efficientnet_dim = 1280
        
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + efficientnet_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Keep original structure to match saved model
        self.dr_classifier = nn.Linear(512, num_classes)
        self.vessel_analyzer = nn.Linear(512, 3)
        self.feature_extractor = nn.Linear(512, 10)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        
        # ViT features
        vit_output = self.vit(x)
        vit_features = vit_output.last_hidden_state[:, 0]
        
        # EfficientNet features
        efficientnet_features = self.efficient_net.extract_features(x)
        efficientnet_features = self.efficient_net._avg_pooling(efficientnet_features)
        efficientnet_features = efficientnet_features.view(efficientnet_features.size(0), -1)
        
        # Feature fusion
        combined_features = torch.cat([vit_features, efficientnet_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # Apply post-processing to vessel metrics
        vessel_output = torch.tanh(self.vessel_analyzer(fused_features))
        
        return {
            'dr_class': self.dr_classifier(fused_features),
            'vessel_metrics': vessel_output,
            'medical_features': self.feature_extractor(fused_features),
            'shared_features': fused_features
        }