import torch
from ..layers.upsampling import upsample_activation
from ..layers.normalization import normalize_activation
from ..layers.masking import apply_mask
from ..scoring.forward_pass import forward_score
from ..scoring.score_normalization import normalize_scores
from ..cam.score_cam import score_cam
from ..layers.activation_extractor import ActivationExtractor

class ScoreCAMPipeline:
    def __init__(self, backbone, target_layer, input_size=(224,224)):
        self.backbone = backbone
        self.target_layer = target_layer
        self.input_size = input_size
        self.extractor = ActivationExtractor(backbone, target_layer)

    def generate_heatmap(self, input_image, target_class=None):
        _ = self.backbone(input_image)
        feature_maps = self.extractor.get_activation()  

        upsampled_maps = upsample_activation(feature_maps, self.input_size)
        norm_maps = normalize_activation(upsampled_maps)

        masked_inputs = torch.stack([apply_mask(input_image, m) for m in norm_maps])

        scores = torch.stack([forward_score(self.backbone, mi, target_class) for mi in masked_inputs])

        alpha_k = normalize_scores(scores)

        heatmap = score_cam(norm_maps, alpha_k)
        return heatmap
