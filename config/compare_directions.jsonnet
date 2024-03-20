local model_key = std.extVar('model');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';


local all_steps_l = [
    common.steps_model_func(model_key, models[model_key], data[dataset])
    for dataset in std.objectFields(data)
];
local model_steps = utils.join_objects([s['model_step'] for s in all_steps_l]);
local data_steps = utils.join_objects([s['data_steps'] for s in all_steps_l]);
local train_steps = utils.join_objects([s['train_steps'] for s in all_steps_l]);

{
    'steps': model_steps + data_steps + train_steps + {
        'analysis': {
            type: 'direction_similarity',
            result_inputs: std.objectKeysValues(train_steps),
            probes: std.map(
                function(key) {"ref": key},
                std.objectFields(train_steps)
            ),
        }
    }
}