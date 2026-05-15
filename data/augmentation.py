import numpy as np
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor


class ImageAugmentation:

    def __init__(
        self,
        n_views=40,
        deterministic=False,
        seed=3407,
        base_transform=None,
        preprocess=None,
        num_workers=8,
    ):
        self.n_views = n_views
        self.deterministic = deterministic
        self.seed = seed
        self.num_workers = num_workers

        self.base_transform = base_transform

        self.preprocess = preprocess

        self.aug = transforms.RandomPerspective(
            distortion_scale=0.5,
            p=1.0,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    def worker(self, args):
        image, idx, sample_id = args

        if self.deterministic:
            torch.manual_seed(self.seed + sample_id * 100000 + idx)
            np.random.seed(self.seed + sample_id * 100000 + idx)

        img = image.copy()
        img = self.aug(img)
        img = self.base_transform(img)
        img = self.preprocess(img)

        return img

    def __call__(self, image, n_views=None, n_types=None, sample_id=0):

        num_views = self.n_views if n_views is None else n_views

        base = self.preprocess(
            self.base_transform(image)
        )

        views = [base]

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            outs = list(
                ex.map(
                    self.worker,
                    [(image, i, sample_id) for i in range(num_views)]
                )
            )

        views.extend(outs)

        return views

    def n_type_views(self, img, n_types=None, n_views=None, sample_id=0):
        return self.__call__(img, n_views=n_views, sample_id=sample_id)
