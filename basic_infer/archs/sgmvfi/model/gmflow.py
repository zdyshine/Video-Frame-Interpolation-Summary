import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .utils import feature_add_position

class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []
        flow_s_macthing = []
        flow_s_prop = []
        transformer_features = []

        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            attn_splits = attn_splits_list[scale_idx]

            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            transformer_features.append(torch.cat([feature0, feature1], 0))

        results_dict.update({'trans_feat': transformer_features})

        return results_dict
