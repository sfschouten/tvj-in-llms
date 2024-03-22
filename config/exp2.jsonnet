local dataset = std.extVar('dataset');
local model_key = std.extVar('model');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';

local pos_prem_data_key = dataset + '-original_pos_prem';
local calibration_data_key = dataset + '-no_prem';
local train_data_config = data[dataset][pos_prem_data_key];
local calibration_data_config = data[dataset][calibration_data_key];

local model_config = models[model_key];

local pos_prem_prefix = common.create_method_prefix_func(pos_prem_data_key, model_key, null);


local steps() =
    local train_steps = utils.join_objects([
        common.usv_method_train_steps_func(pos_prem_data_key, model_key, layer)
        for layer in model_config['layers']
    ]) + utils.join_objects([
        common.sv_method_train_steps_func(pos_prem_data_key, model_key, layer)
        for layer in model_config['layers']
    ]);
    local training_data = common.data_gen_steps_func(
        pos_prem_data_key, train_data_config, model_key, model_config,
        {"ref": model_key}, {"ref": model_key + "-tokenizer"}
    );
    local calibration_data = common.data_gen_steps_func(
        calibration_data_key, calibration_data_config, model_key, model_config,
        {"ref": model_key}, {"ref": model_key + "-tokenizer"}
    );

//    local layers_list = [[5,6,7,8,9]];
    local layers_list = [[11,12,13,14,15]];
    local intervened_outputs =  {
        ["INTERVENED_on"+std.toString(layers)+"_with[" + std.strReplace(train_step['key'], '|', ',') + "]"]: {
            type: "generate_with_intervention",
            model: {ref: model_key},
            tokenizer: {ref: model_key + "-tokenizer"},
            dataset: {ref: pos_prem_prefix + 'data'},
            batch_size: model_config['batch_size'],
            all_layers: true,
            model_type: model_config['type'],
            probe: {ref: train_step['key']},
            module_template: model_config['layer_template'],
            layers: layers,
            intervene_on_period: true,
            intervention_sign: -1,
        }
        for layers in layers_list
        for train_step in std.objectKeysValues(train_steps)
        if std.member(train_step['key'], 'layer'+layers[2])   # use direction found in (first) intervention layer
        if std.member(train_step['key'], 'lm_head_baseline') == false  # skip LM-Head baseline (no truth-direction)
    };

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

    # evaluate using the probes we already trained prior to intervention and evaluate on intervened hidden states
    local intervened_eval_steps = {
        [int_output['key'] + "|" + std.split(train_step['key'], '|')[2] + "|eval"]: {
            type: 'eval_belief_probe',
            data: {ref: int_output['key'] + "|" + std.split(train_step['key'], '|')[2] + "|normalized_hidden_states"},
            probe: {ref: train_step['key']}  # this is already a {ref: ...}
        }
        for int_output in std.objectKeysValues(intervened_outputs)
        for train_step in std.objectKeysValues(train_steps)
        if std.member(int_output['key'], std.split(train_step['key'], '|')[4])  # same method
    };

    local original_eval_steps = {
        [std.strReplace(train_step['key'], 'train', 'eval')]: {
            type: 'eval_belief_probe',
            data: train_step['value']['train_data'],
            probe: {ref: train_step['key']},
        }
        for train_step in std.objectKeysValues(train_steps)
    };
    local eval_steps = intervened_eval_steps + original_eval_steps;

    {
        [model_key]: {
            type: "transformers::AutoModelForCausalLM::from_pretrained::step",
            pretrained_model_name_or_path: model_config['key'],
            device_map: {"": "cuda:0"},
            torch_dtype: "float16",
            revision: model_config['revision'],
        },
        [model_key + "-tokenizer"]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model_config['key'],
        }
    } + training_data + calibration_data + train_steps + intervened_outputs + intervened_hidden_states + eval_steps
      + {
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
    };

{
    "steps": steps()
}
