local dataset_key = std.extVar('dataset');
local model_key = std.extVar('model');

local datasets = import 'data.libsonnet';
local models = import 'models.libsonnet';
local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';

local model_config = models[model_key];
local dataset_config = datasets[dataset_key];

local model_step = common.model_and_tokenizer_func(model_key, model_config);

{
    "steps": model_step + utils.join_objects([
        common.data_gen_steps_func(
            data['key'], data['value'], model_key, model_config, {"ref": model_key}, {"ref": model_key + "-tokenizer"}
        )
        for data in std.objectKeysValues(dataset_config)
    ])
}

