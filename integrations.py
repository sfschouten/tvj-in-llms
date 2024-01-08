import random
from typing import Type, Optional

from tango.common import Registrable
from tango.integrations.torch.model import Model
from tango.integrations.transformers import Config

import torch
import numpy as np

from transformers.utils.quantization_config import QuantizationConfigMixin
from transformers.models.auto import modeling_auto
from transformers import GPTQConfig as GPTQConfigOriginal


# QUANTIZATION


class QuantizationConfig(QuantizationConfigMixin, Registrable):
    def __init__(self, *args, **kwargs):
        super(QuantizationConfigMixin).__init__(*args, **kwargs)


QuantizationConfig.register('gptq-config')(GPTQConfigOriginal)
# TODO register other quantization configs


# override default `from_pretrained` wrappers


def auto_model_wrapper_factory(cls: type) -> Type[Model]:
    class AutoModelWrapper(cls, Model):  # type: ignore
        @classmethod
        def from_pretrained(
                cls, pretrained_model_name_or_path: str, torch_dtype: str, config: Optional[Config] = None,
                quantization_config: Optional[QuantizationConfig] = None, **kwargs
        ) -> Model:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            model = super().from_pretrained(
                pretrained_model_name_or_path, torch_dtype=getattr(torch, torch_dtype), config=config,
                quantization_config=quantization_config, **kwargs
            )
            model.__deepcopy__ = lambda x: x
            return model

        @classmethod
        def from_config(cls, config: Config, **kwargs) -> Model:
            return super().from_config(config, **kwargs)

    return AutoModelWrapper


for name, cls in modeling_auto.__dict__.items():
    if isinstance(cls, type) and name.startswith("AutoModel"):
        wrapped_cls = auto_model_wrapper_factory(cls)
        Model.register(
            "transformers::" + name + "::from_pretrained", constructor="from_pretrained", exist_ok=True
        )(wrapped_cls)
        Model.register(
            "transformers::" + name + "::from_config", constructor="from_config", exist_ok=True
        )(wrapped_cls)

