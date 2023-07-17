import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super(MultimodalModel, self).__init__()

        fused_representation_size = 512
        image_fused_representation_size = 512
        text_fused_representation_size = 100
        num_attention_heads = 4

        self.image_model = image_model
        self.text_model = text_model
        self.num_classes = num_classes

        self.image_attention = nn.MultiheadAttention(image_fused_representation_size, num_attention_heads)
        self.image_attention_Linear = nn.Linear(image_fused_representation_size, fused_representation_size)
        self.text_attention = nn.MultiheadAttention(text_fused_representation_size, num_attention_heads)
        self.text_attention_Linear = nn.Linear(text_fused_representation_size, fused_representation_size)
        self.fusion_layer = nn.Linear(image_model.fc1.out_features + text_model.lstm1.hidden_size * 2, fused_representation_size)
        self.nonlinear_layer = nn.Linear(fused_representation_size, fused_representation_size)
        self.classification_layer = nn.Linear(fused_representation_size, num_classes)

    def forward(self, text_inputs, image_inputs):
        image_features = self.image_model(image_inputs)
        text_features = self.text_model(text_inputs)


        # Attention weights
        image_attention_weights, _ = self.image_attention(image_features, image_features, image_features)
        text_attention_weights, _ = self.text_attention(text_features, text_features, text_features)

        image_attention_weights = self.image_attention_Linear(image_attention_weights)
        text_attention_weights = self.text_attention_Linear(text_attention_weights)

        # Softmax attention weights
        image_attention_weights = F.softmax(image_attention_weights, dim=1)
        text_attention_weights = F.softmax(text_attention_weights, dim=1)

        # Fusion representation of the modalities
        fused_features = torch.cat((image_features, text_features), dim=1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.nonlinear_layer(fused_features)

        # Weighted sum of fused features using attention weights
        weighted_features = image_attention_weights * fused_features + text_attention_weights * fused_features

        # Remove the batch dimension
        weighted_features = weighted_features.squeeze(0)

        # Classification
        output = self.classification_layer(weighted_features)
        probabilities = F.softmax(output, dim=-1)

        return probabilities
    
