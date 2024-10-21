import copy

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from typing import Dict, Callable, List

from module_attr_mixin import ModuleAttrMixin




def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class SimpleRGBObsEncoder(nn.Module):
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 n_emb: int = 768,
                 pretrained: bool = False,
                 frozen: bool = False,
                 use_group_norm: bool = True,
                 feature_aggregation: str = "cls",
                 downsample_ratio: int = 32):
        """
        Assumes rgb input: B, T, C, H, W
        For images of fixed size 224x224
        """
        super().__init__()

        self.feature_aggregation = feature_aggregation

        # Load the model
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classification layer
            global_pool=''  # no global pooling
        )
        
        if frozen:
            assert pretrained, "If frozen, the model should be pretrained."
            for param in model.parameters():
                param.requires_grad = False

        # Handling ResNet model specific downsample ratio
        if model_name.startswith('resnet'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        else:
            # For models like ViT
            feature_dim = model.embed_dim if hasattr(model, 'embed_dim') else n_emb

        # Optional GroupNorm replacement if not pretrained
        if use_group_norm and not pretrained:
            model = self.replace_bn_with_gn(model)

        self.model = model
        self.n_emb = n_emb
        self.feature_dim = feature_dim

        # Optional projection if feature size does not match n_emb
        self.projection = nn.Identity()
        if feature_dim != n_emb:
            self.projection = nn.Linear(in_features=feature_dim, out_features=n_emb)

    def replace_bn_with_gn(self, model):
        """Replace all BatchNorm layers with GroupNorm"""
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                gn_layer = nn.GroupNorm(
                    num_groups=(module.num_features // 16) if (module.num_features % 16 == 0) else (module.num_features // 8),
                    num_channels=module.num_features
                )
                setattr(model, name, gn_layer)
        return model

    def aggregate_feature(self, feature):
        """Aggregate features, handling different feature aggregation strategies"""
        if self.feature_aggregation == 'cls':
            return feature[:, 0, :]  # ViT uses the CLS token for classification
        else:
            return feature  # Return all tokens or raw feature map

    def forward(self, img):
        """
        img: B, T, C, H, W
        """
        B, T, C, H, W = img.shape
        assert H == W == 224, "Input image must be 224x224"
        img = img.reshape(B * T, C, H, W)

        # Pass through the model
        raw_feature = self.model(img)
        feature = self.aggregate_feature(raw_feature)

        # Apply projection if necessary
        emb = self.projection(feature)

        # Reshape to B, T, n_emb
        emb = emb.view(B, T, self.n_emb)
        return emb

    @torch.no_grad()
    def output_shape(self):
        example_img = torch.zeros((1, 1, 3, 224, 224))
        example_output = self.forward(example_img)
        return example_output.shape

