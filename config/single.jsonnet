local dataset = std.extVar('dataset');
local model_key = std.extVar('model');
local layer = std.extVar('layer');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';


local data_key = dataset + '-no_prem';

local train_steps = common.sv_method_train_steps_func(data_key, model_key, layer);

local eval_steps = {
    [std.strReplace(probe_obj['key'], 'train', 'eval@' + data_key)]: {
        "type": "eval_belief_probe",
        data: {
            "ref": common.create_method_prefix_func(data_key, model_key, null)
               + std.join('|', std.split(probe_obj['value']['train_data']['ref'], '|')[2:4])
        },
        probe: {"ref": probe_obj['key']},
    }
    for probe_obj in std.objectKeysValues(train_steps)
};


{
    'steps': common.model_and_tokenizer_func(model_key, models[model_key])
           + common.norm_data_steps_func(data_key, data[dataset][data_key], model_key, models[model_key],
                                        {ref: model_key}, {ref: model_key + '-tokenizer'})
           + train_steps + eval_steps
}