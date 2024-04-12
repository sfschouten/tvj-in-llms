local utils = import 'utils.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';

# dummy model, not used
local model_key = 'llama2-7b';
local model_config = models[model_key];

local prompts = {
    'lcb_snli': import '../beliefprobing/datasets/lcb_snli/templates.libsonnet',
    'lcb_ent_bank': import '../beliefprobing/datasets/lcb_ent_bank/templates.libsonnet',
};
local all_variants = utils.join_objects([data[dataset] for dataset in std.objectFields(prompts)]);


{
    "steps": {
        [model_key + "-tokenizer"]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model_config['key'],
        },
    } + {
        ['data_' + data_kv['key']]: {
            type: 'load_data',
            dataset_name_or_path: data_kv['value']['name'],
            dataset_config_name: data_kv['value']['config'],
            prompt_template: prompts[data_kv['value']['name']][data_kv['value']['prompt']],
            split: data_kv['value']['split'],
            tokenizer: {ref: model_key + "-tokenizer"},
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