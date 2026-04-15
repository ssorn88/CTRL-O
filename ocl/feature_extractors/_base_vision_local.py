from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image


def unpack_tuple(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result
    return wrapper


class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...


@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad = int((max_wh - w) / 2)
        vertical_pad = int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


class VisionBackbone(nn.Module, ABC):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier = vision_backbone_id
        self.image_resize_strategy = image_resize_strategy
        self.default_image_size = default_image_size
        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]:
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def num_patches(self) -> int:
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        ...