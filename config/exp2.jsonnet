local dataset = std.extVar('dataset');
local model_key = std.extVar('model');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';

# the training data for the probes whose direction (theta) we use for the intervention
local train_data_key = dataset + '-original_pos_prem';
//local train_data_key = dataset + '-no_prem';
local train_data_config = data[dataset][train_data_key];

# which data to intervene on
local intervention_data_key = dataset + '-original_pos_prem';
local intervention_data_config = data[dataset][intervention_data_key];
local intervention_data_prefix = common.create_method_prefix_func(intervention_data_key, model_key, null);

# the training data for the probes with which we evaluate the effect of the intervention
local eval_data_key = dataset + '-original_pos_prem';
local eval_data_config = data[dataset][eval_data_key];



#
local model_config = models[model_key];
local model_and_tokenizer = common.model_and_tokenizer_func(model_key, model_config);


local calibration_data_key = dataset + '-no_prem';
local calibration_data_config = data[dataset][calibration_data_key];
local calibration_data = common.norm_data_steps_func(
    calibration_data_key, calibration_data_config, model_key, model_config,
    {"ref": model_key}, {"ref": model_key + "-tokenizer"}
);

# probes whose theta we use to intervene
local training_data = common.norm_data_steps_func(
    train_data_key, train_data_config, model_key, model_config, {"ref": model_key}, {"ref": model_key + "-tokenizer"}
);
local int_probe_train_steps = utils.join_objects([
    common.usv_method_train_steps_func(train_data_key, model_key, layer)
    for layer in model_config['layers']
]) + utils.join_objects([
    common.sv_method_train_steps_func(train_data_key, model_key, layer)
    for layer in model_config['layers']
]);

# probes whose we evaluate with post intervention
local eval_data = common.norm_data_steps_func(
    eval_data_key, eval_data_config, model_key, model_config, {"ref": model_key}, {"ref": model_key + "-tokenizer"}
);
local eval_probe_train_steps = utils.join_objects([
    common.usv_method_train_steps_func(eval_data_key, model_key, layer)
    for layer in model_config['layers']
]) + utils.join_objects([
    common.sv_method_train_steps_func(eval_data_key, model_key, layer)
    for layer in model_config['layers']
]);


local intervention_data = common.norm_data_steps_func(
    intervention_data_key, intervention_data_config, model_key, model_config,
    {"ref": model_key}, {"ref": model_key + "-tokenizer"}
);

local layers_list = [[10, 11, 12, 13, 14, 15, 16]];
//local layers_list = [[8, 9, 10, 11, 12, 13, 14]];
//local layers_list = [[6, 7, 8, 9, 10, 11, 12]];
local intervened_outputs =  {
    ["INTERVENED_on"+std.toString(layers)+"_with[" + std.strReplace(train_step['key'], '|', ',') + "]"]: {
        type: "generate_with_intervention",
        model: {ref: model_key},
        tokenizer: {ref: model_key + "-tokenizer"},
        dataset: {ref: intervention_data_prefix + 'data'},
        batch_size: model_config['batch_size'],
        all_layers: true,
        model_type: model_config['type'],
        probes: [
//            {ref: std.strReplace(train_step['key'], 'layer1', 'layer'+l)}
            {ref: std.strReplace(train_step['key'], 'layer1', 'layer'+layers[6])}
            for l in layers
        ],
        module_template: model_config['layer_template'],
        layers: layers,
        intervene_on_answer: true,
        intervene_on_period: true,
        intervention_sign: -1,
    }
    for layers in layers_list
    for train_step in std.objectKeysValues(int_probe_train_steps)
    if std.member(train_step['key'], '|layer1|')
    if std.member(train_step['key'], 'lm_head_baseline') == false  # skip LM-Head baseline (no truth-direction)
};

# split and normalize post-intervention hidden states
local intervened_hidden_states = {
    [int_output['key'] + '|layer' + layer + '|split_outputs']: {
        "type": "create_splits",
        gen_out: {ref: int_output['key']},
        layer_index: layer
    }
    for int_output in std.objectKeysValues(intervened_outputs)
    for layer in model_config['layers']
} + {
    [int_output['key'] + '|layer' + layer + '|normalized_hidden_states']: {
        "type": "normalize",
        data: {ref: int_output['key'] + '|layer' + layer + "|split_outputs"},
        var_normalize: false,
    }
    for int_output in std.objectKeysValues(intervened_outputs)
    for layer in model_config['layers']
};

# evaluate
local intervened_eval_steps = {
    [int_output['key'] + "|" + std.split(train_step['key'], '|')[2] + "|eval_with["
     + std.split(train_step['key'], '|')[4] + "]"
    ]: {
        type: 'eval_belief_probe',
        data: {ref: int_output['key'] + "|" + std.split(train_step['key'], '|')[2] + "|normalized_hidden_states"},
        probe: {ref: train_step['key']}
    }
    for int_output in std.objectKeysValues(intervened_outputs)
    for train_step in std.objectKeysValues(eval_probe_train_steps)
    if std.member(int_output['key'], std.split(train_step['key'], '|')[4])  # same method
//    || std.split(train_step['key'], '|')[4] == 'lm_head_baseline'
};
local original_eval_steps = {
    [std.strReplace(train_step['key'], 'train', 'eval')]: {
        type: 'eval_belief_probe',
        data: {ref: intervention_data_prefix + std.split(train_step['key'], '|')[2] + "|normalized_hidden_states"},
        probe: {ref: train_step['key']},
    }
    for train_step in std.objectKeysValues(eval_probe_train_steps)
    if std.member(train_step['key'], 'lm_head_baseline') == false  # skip LM-Head baseline (no truth-direction)
};
local eval_steps = intervened_eval_steps + original_eval_steps;


{
    "steps": model_and_tokenizer + training_data + calibration_data + intervention_data + int_probe_train_steps
    + eval_probe_train_steps + intervened_outputs + intervened_hidden_states + eval_steps + {
        'results_db': {
            type: "duckdb_builder2",
            result_inputs: std.objectKeysValues(eval_steps),
            results: std.map(
                function(key) {"ref": key},
                std.objectFields(eval_steps)
            ),
        },
        'analysis': {
            type: 'causal_analysis',
            results_db: {ref: 'results_db'}
        }
    }
}
