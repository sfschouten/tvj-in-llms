local dataset_key = std.extVar('dataset');
local model_key = std.extVar('model');

local datasets = import 'data.libsonnet';
local models = import 'models.libsonnet';
local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';



local model_config = models[model_key];
local dataset_config = datasets[dataset_key];


local dataset_name = std.split(std.objectFields(dataset_config)[0], '-')[0];

# load model
local model_step = common.model_and_tokenizer_func(model_key, model_config);


local data_gen_steps(data_key, data_config, model_key, model_config, model_object, tokenizer_object) =
    // a prefix for the steps that involve this data and this model
    local prefix = data_key + "|" + model_key + "|";

    {
        [prefix + "data"]: {
            type: "load_data",
            dataset_name: data_config['name'],
            split: data_config['split'],
            tokenizer: tokenizer_object,
            dataset_config_name: data_config['config'],
            prompt_name: data_config['prompt'],
            model_type: model_config['type'],
        },
        [prefix + "outputs"]: {
            type: "generate_hidden_states",
            model: model_object,
            tokenizer: tokenizer_object,
            dataset: {"ref": prefix+"data"},
            batch_size: model_config['batch_size'],
            all_layers: true,
            model_type: model_config['type'],
        }
    };


{
    "steps": model_step + utils.join_objects([
        data_gen_steps(data['key'], data['value'], model_key, model_config, {"ref": model_key}, {"ref": model_key + "-tokenizer"})
        for data in std.objectKeysValues(dataset_config)
    ])
}

