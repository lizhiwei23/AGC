import clip
import torch
import torch.nn.functional as F
from torchvision import transforms

from data.augmentation import ImageAugmentation
from utils.pull_modes import build_tangent_filtered_feature, step_feature

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class Adaptive_geo_cor:
    def __init__(
        self,
        clip_model,
        all_classes,
        text_features=None,
        prompt_style="default",
        n_views=32,
        beta_clean=0.45,
        beta_adv=2.25,
        device="cpu",
    ):
        self.device = device
        self.model = clip_model.to(device).eval()
        self.all_classes = all_classes
        self.n_views = 32 if n_views is None else max(0, int(n_views))
        self.beta_clean = float(beta_clean)
        self.beta_adv = float(beta_adv)
        
        self.augmentation = ImageAugmentation(
            n_views=self.n_views,
            deterministic=False,
            base_transform=transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                ]
            ),
            preprocess=transforms.Compose([transforms.ToTensor()]),
        )
        self.clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.build_pulled_feature = build_tangent_filtered_feature

        if text_features is not None:
            self.text_features = F.normalize(text_features.to(device), dim=-1)
        else:
            if prompt_style == "rtpt":
                prompts = [f"a photo of a {c.replace('_', ' ')}." for c in all_classes]
            else:
                prompts = [f"a photo of a {c}" for c in all_classes]

            texts = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                self.text_features = F.normalize(self.model.encode_text(texts), dim=-1)

    def _encode_base_and_augmented_images(self, pil_image):
        views = self.augmentation(pil_image)

        if isinstance(views, list):
            views = torch.stack(views, dim=0)
        all_imgs = views.to(self.device, non_blocking=True)
        all_imgs = (all_imgs - self.clip_mean) / self.clip_std
        with torch.no_grad():
            feats_tensor = F.normalize(self.model.encode_image(all_imgs).float(), dim=-1)

        if feats_tensor.size(0) == 1:
            return feats_tensor[0:1], feats_tensor[0:1]

        return feats_tensor[0:1], feats_tensor[1:]

    def _predict_from_feature(self, defense_feats, logit_scale=None, text_features_t=None):
        mean_tensor = defense_feats.view(1, -1).float()
        
        if logit_scale is None:
            logit_scale = self.model.logit_scale.exp().float()
        if text_features_t is None:
            text_features_t = self.text_features.T.float()
        else:
            text_features_t = text_features_t.float()

        logit_scale = logit_scale.to(device=mean_tensor.device, dtype=mean_tensor.dtype)
        text_features_t = text_features_t.to(device=mean_tensor.device, dtype=mean_tensor.dtype)
        logits = logit_scale * (mean_tensor @ text_features_t)
        return int(logits.argmax(dim=-1).item())
    

    def predict_pil(self, pil_image):
        base_feat, aug_feats = self._encode_base_and_augmented_images(pil_image)
        defense_feats = self.build_pulled_feature(base_feat, aug_feats, beta_clean=self.beta_clean, beta_adv=self.beta_adv)
        return self._predict_from_feature(defense_feats)