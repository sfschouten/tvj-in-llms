import random
import os
import uuid

from pathlib import Path
from typing import Type, Optional, Dict

import duckdb
from duckdb import DuckDBPyConnection
from tango import Format, Step
from tango.common import Registrable, PathOrStr
from tango.integrations.torch.model import Model
from tango.integrations.transformers import Config, Tokenizer

import torch
import numpy as np

from transformers.utils.quantization_config import QuantizationConfigMixin
from transformers.models.auto import modeling_auto
from transformers import GPTQConfig as GPTQConfigOriginal
from transformers import AutoTokenizer

from promptsource.templates import Template
from promptsource.templates import DatasetTemplates


# QUANTIZATION

class QuantizationConfig(QuantizationConfigMixin, Registrable):
    def __init__(self, *args, **kwargs):
        super(QuantizationConfigMixin).__init__(*args, **kwargs)


QuantizationConfig.register('gptq-config')(GPTQConfigOriginal)

# TODO register other quantization configs


# Override default `from_pretrained` wrappers

def auto_model_wrapper_factory(cls: type) -> tuple[Type[Model], Type[Step[Model]]]:
    class AutoModelWrapper(cls, Model):  # type: ignore
        @classmethod
        def from_pretrained(
                cls, pretrained_model_name_or_path: str, torch_dtype: str = 'auto', config: Optional[Config] = None,
                quantization_config: Optional[QuantizationConfig] = None, **kwargs
        ) -> Model:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            if torch_dtype != 'auto':
                torch_dtype = getattr(torch, torch_dtype)

            model = super().from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype, config=config,
                quantization_config=quantization_config, **kwargs
            )

            def new_deepcopy(self, _):
                return self

            model.__class__.__deepcopy__ = new_deepcopy
            return model

        @classmethod
        def from_config(cls, config: Config, **kwargs) -> Model:
            return super().from_config(config, **kwargs)

    # dummy step to load model once and use in multiple steps, and only when needed by another step
    class AutoModelLoaderPretrained(Step):
        CACHEABLE = False
        MODEL_CLASS = AutoModelWrapper
        SKIP_ID_ARGUMENTS = {'torch_dtype', 'trust_remote_code'}
        SKIP_DEFAULT_ARGUMENTS = {'torch_dtype', 'trust_remote_code'}

        def run(self, **kwargs) -> Model:
            return self.MODEL_CLASS.from_pretrained(**kwargs)

    return AutoModelWrapper, AutoModelLoaderPretrained


for name, cls in modeling_auto.__dict__.items():
    if isinstance(cls, type) and name.startswith("AutoModel"):
        wrapper_cls, loader_cls = auto_model_wrapper_factory(cls)
        name_prefix = "transformers::" + name + "::"

        Model.register(name_prefix + "from_pretrained", constructor="from_pretrained", exist_ok=True)(wrapper_cls)
        Model.register(name_prefix + "from_config", constructor="from_config", exist_ok=True)(wrapper_cls)

        Step.register(name_prefix + "from_pretrained::step")(loader_cls)


# same dummy step but for tokenizer
@Step.register('transformers::AutoTokenizer::from_pretrained::step')
class AutoTokenizerLoader(Step):
    CACHEABLE = False
    SKIP_ID_ARGUMENTS = {'trust_remote_code'}

    def run(self, **kwargs) -> Tokenizer:
        return AutoTokenizer.from_pretrained(**kwargs)


#  CUSTOM FORMATS


class TupleFormat(Format[tuple]):

    def __init__(self, formats: tuple[Format, ...]):
        self.formats = formats

    def write(self, artifact: tuple, dir: PathOrStr):
        for i, (elem, format) in enumerate(zip(artifact, self.formats)):
            subdir = Path(dir) / f'elem_{i}'
            subdir.mkdir(exist_ok=True)
            format.write(elem, subdir)

    def read(self, dir: PathOrStr) -> tuple:
        result = [None] * len(self.formats)
        for entry in os.listdir(dir):
            subdir = Path(dir) / entry
            if not os.path.isdir(subdir) or not entry.startswith('elem_'):
                continue

            i = int(entry.split('_')[1])
            fmt = self.formats[i]
            result[i] = fmt.read(subdir)

        return tuple(result)


class DuckDBFormat(Format[DuckDBPyConnection]):

    def write(self, artifact: DuckDBPyConnection, dir: PathOrStr):
        artifact.sql(f"EXPORT DATABASE '{str(dir)}';")

    def read(self, dir: PathOrStr) -> DuckDBPyConnection:
        con = duckdb.connect()
        con.sql(f"IMPORT DATABASE '{str(dir)}';")
        return con


#  PROMPT TEMPLATES


class PromptTemplate(Registrable, Template):
    pass


class NativePromptsourceTemplateLoader:

    @staticmethod
    def load_existing_template(dataset_name: str, prompt_name: str, prompt_i: Optional[int] = None):
        all_prompts = DatasetTemplates(dataset_name)

        prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        if prompt_i is not None:
            prompt_name = prompt_name_list[prompt_i]

        return all_prompts[prompt_name]


PromptTemplate.register(
    "promptsource_template", constructor="load_existing_template", exist_ok=True
)(NativePromptsourceTemplateLoader)


@PromptTemplate.register('custom_template')
class CustomPromptTemplate(Template):
    NAMESPACE = uuid.UUID('2649977f-b127-4e09-a6ae-64881baa8f4c')

    def __init__(
            self, name: str, jinja: str, answer_choices: str,
            metadata_original_task: Optional[bool] = None, metadata_choices_in_prompt: Optional[bool] = None,
            metadata_metrics: Optional[list[str]] = None, metadata_languages: Optional[list[str]] = None,
            reference: Optional[str] = '',
    ):
        metadata = Template.Metadata(
            original_task=metadata_original_task,
            choices_in_prompt=metadata_choices_in_prompt,
            metrics=metadata_metrics,
            languages=metadata_languages,
        )
        super().__init__(name, jinja, reference, metadata, answer_choices)
        uuid_name = name + jinja + answer_choices + str(metadata_original_task) + str(metadata_choices_in_prompt) \
            + str(metadata_metrics) + str(metadata_languages) + str(reference)
        self.id = uuid.uuid5(self.NAMESPACE, uuid_name)
