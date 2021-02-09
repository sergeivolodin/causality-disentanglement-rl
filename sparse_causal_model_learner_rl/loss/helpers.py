import torch


def gather_additional_features(additional_feature_keys, **kwargs):
    """Get tensor of additional features."""
    if additional_feature_keys:
        add_features_y = torch.cat([kwargs[k] for k in additional_feature_keys], dim=1)
        return add_features_y
    return None