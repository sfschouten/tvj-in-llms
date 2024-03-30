local utils = import 'utils.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';


# dummy model, not used
local model_key = 'llama2-7b';
local model_config = models[model_key];

local all_variants = data['snli'] + data['entbank'];

{
    "steps": {
        [model_key + "-tokenizer"]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model_config['key'],
        },
    } + {
        ['data_' + data_kv['key']]: {
            type: 'load_data',
            dataset_name: data_kv['value']['name'],
            split: data_kv['value']['split'],
            tokenizer: {ref: model_key + "-tokenizer"},
            dataset_config_name: data_kv['value']['config'],
            prompt_name: data_kv['value']['prompt'],
            model_type: model_config['type'],
        }
        for data_kv in std.objectKeysValues(all_variants)
    } + {
        ['extract_samples_' + data_kv['key']]: {
            type: 'extract_data_samples',
            dataset: {ref: 'data_' + data_kv['key']}
        }
        for data_kv in  std.objectKeysValues(all_variants)
    }
}