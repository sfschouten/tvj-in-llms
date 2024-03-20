local utils = import 'utils.libsonnet';

local create_method_prefix(data_key, model_key, layer) =
    if layer == null then
        data_key + "|" + model_key + "|"
    else
        data_key + "|" + model_key + "|layer" + layer + "|";


local data_gen_steps(data_key, data_config, model_key, model_config, model_object, tokenizer_object) =
    // a prefix for the steps that involve this data and this model
    local prefix = create_method_prefix(data_key, model_key, null);

    {
	    [prefix + "data"]: {
	        "type": "load_data",
            dataset_name: data_config['name'],
            split: data_config['split'],
            tokenizer: tokenizer_object,
            dataset_config_name: data_config['config'],
            prompt_name: data_config['prompt'],
            model_type: model_config['type'],
            add_period: true,
	    },
	    [prefix + "outputs"]: {
	        "type": "generate_hidden_states",
	        model: model_object,
	        tokenizer: tokenizer_object,
	        dataset: {"ref": prefix+"data"},
	        batch_size: model_config['batch_size'],
            all_layers: true,
            model_type: model_config['type'],
	    },
//	    [prefix + 'hidden_state_ranks']: {
//	        "type": "print_rank",
//	        gen_out: {"ref": prefix+"outputs"},
//	    },
    } + {
	    [prefix + 'layer' + layer + '|split_outputs']: {
	        "type": "create_splits",
	        gen_out: {"ref": prefix+"outputs"},
	        layer_index: layer
	    } for layer in model_config['layers']
	} + {
	    [prefix + 'layer' + layer  + '|normalized_hidden_states']: {
	        "type": "normalize",
	        data: {"ref": prefix + 'layer' + layer + "|split_outputs"},
	        var_normalize: false,
	    } for layer in model_config['layers']
	} + {
	    [prefix + 'layer' + layer + '|var_normalized_hidden_states']: {
	        "type": "normalize",
	        data: {"ref": prefix + 'layer' + layer + "|split_outputs"},
	        var_normalize: true,
	    } for layer in model_config['layers']
	};


local CCS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
//local CCS_SEEDS = [0, 1];


// unsupervised methods
local usv_method_train_steps(data_key, model_key, layer) =
    local prefix = create_method_prefix(data_key, model_key, layer);

    local cal_data = std.split(data_key, '-')[0] + '-no_prem';
    local calibration_prefix = create_method_prefix(cal_data, model_key, layer);

    local ccs_trials = {
        [prefix + 'trial|ccs_' + seed]: {
            "type": "train_belief_probe",
            train_data: {"ref": prefix+"normalized_hidden_states"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probe: {
                "type": "ccs_gd",
                seed: seed,
                n_epochs: 1000,
                lr: 1e-3,
                batch_size: -1,     # full batch
                weight_decay: 0.01,
                use_constraint: false,
                informative_loss: 'min_sq'
            }
        }
        for seed in CCS_SEEDS
    };

    {
        [prefix + 'train|' + 'lm_head_baseline_calibrate=' + calibrate]: {
            "type": "train_belief_probe",
            train_data: {"ref": prefix+"split_outputs"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probe: {
                "type": "lm_head_baseline",
                calibrate: calibrate
            },
        }
        for calibrate in [true] #, false]
    } + {
        [prefix + 'train|' + 'ccr']: {
            "type": "train_belief_probe",
            train_data: {"ref": prefix+"normalized_hidden_states"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probe: {"type": "ccr", seed: 0},
        }
    } + ccs_trials + {
        [prefix + 'train|ccs']: {
            "type": "select_best",
            train_data: {"ref": prefix+"normalized_hidden_states"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probes: [
                {"ref": prefix + 'trial|' + 'ccs_' + seed}
                for seed in CCS_SEEDS
            ],
            probe_configs: std.objectKeysValues(ccs_trials)
        }
    } + {
//        [prefix + 'train|' + 'ccs_linear-prio=' + prio]: {
//            "type": "train_belief_probe",
//            data: {"ref": prefix+"normalized_hidden_states"},
//            probe: {
//                "type": "ccs_linear",
//                priority: prio
//            },
//        }
//        for prio in [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
//    } + {
//        [prefix + 'train|' + 'unsupervised-massmean']: {
//            "type": "train_belief_probe",
//            train_data: {"ref": prefix+"normalized_hidden_states"},
//            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
//            probe: {"type": "unsupervised_mass_mean", linkage: "complete"},
//        }
    };


// supervised methods
local sv_method_train_steps(data_key, model_key, layer) =
    local prefix = create_method_prefix(data_key, model_key, layer);

    local cal_data = std.split(data_key, '-')[0] + '-no_prem';
    local calibration_prefix = create_method_prefix(cal_data, model_key, layer);

    {
        [prefix + 'train|' + 'logistic_baseline']: {
            "type": "train_belief_probe",
            train_data: {"ref": prefix+"normalized_hidden_states"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probe: {"type": "lr_sklearn"},
        }
    } + {
        [prefix + 'train|' + 'massmean-iid=' + iid]: {
            "type": "train_belief_probe",
            train_data: {"ref": prefix+"normalized_hidden_states"},
            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
            probe: {
                "type": "mass_mean",
                iid: iid
            },
        }
        for iid in [false, true]
    };


// function that returns all steps for a given model and dataset
local steps_model(model_key, model_config, dataset_config) =
    local dataset_name = std.split(std.objectFields(dataset_config)[0], '-')[0];

    # load model
    local model_step = {
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
    };

    local join_objects(objs) = std.foldl(function(acc, obj) acc + obj, objs, {});

    local original_pos_prem = dataset_name + '-original_pos_prem';

    # load data and obtain hidden states
    local data_steps = join_objects([
        data_gen_steps(data['key'], data['value'], model_key, model_config,
                        {"ref": model_key}, {"ref": model_key + "-tokenizer"})
        for data in std.objectKeysValues(dataset_config)
    ]);

    # train probing methods
    local usv_train_steps_singles = join_objects([
        usv_method_train_steps(data_key, model_key, layer)
        for layer in model_config['layers']
        for data_key in std.objectFields(dataset_config) if data_key != original_pos_prem
    ]);
    local usv_train_steps_pos_prem = join_objects([
        usv_method_train_steps(original_pos_prem, model_key, layer)
        for layer in model_config['layers']
    ]);
    //local usv_train_steps_combined = usv_method_train_steps(...); TODO

    local sv_train_steps = join_objects([
        sv_method_train_steps(original_pos_prem, model_key, layer)
        for layer in model_config['layers']
    ]);

    # evaluate unsupervised methods on same data variant they were trained on
    local usv_eval_steps_singles = {
        [std.strReplace(probe_obj['key'], 'train', 'eval')]: {
            "type": "eval_belief_probe",
            data: probe_obj['value']['train_data'],
            probe: {"ref": probe_obj['key']},
        }
        for probe_obj in std.objectKeysValues(usv_train_steps_singles)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0    # skip individual trials
    };
    # evaluate unsupervised methods trained on pos_prem on all other data variants
    local usv_eval_pos_prem_steps = {
        [std.strReplace(probe_obj['key'], 'train', 'eval@' + data_key)]: {
            "type": "eval_belief_probe",
            data: {
                "ref": create_method_prefix(data_key, model_key, null)
                   + std.join('|', std.split(probe_obj['value']['train_data']['ref'], '|')[2:4])
            },
            probe: {"ref": probe_obj['key']},
        }
        for probe_obj in std.objectKeysValues(usv_train_steps_pos_prem)
        for data_key in std.objectFields(dataset_config)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0    # skip individual trials
    };
    # evaluate unsupervised methods trained on combined data on all data variants
    # TODO

    # evaluate supervised methods trained on pos_prem on all data variants
    local sv_eval_steps = {
        [std.strReplace(probe_obj['key'], 'train', 'eval@' + data_key)]: {
            "type": "eval_belief_probe",
            data: {
                "ref": create_method_prefix(data_key, model_key, null)
                   + std.join('|', std.split(probe_obj['value']['train_data']['ref'], '|')[2:4])
            },
            probe: {"ref": probe_obj['key']},
        }
        for probe_obj in std.objectKeysValues(sv_train_steps)
        for data_key in std.objectFields(dataset_config)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0    # skip individual trials
    };

    {
        "model_step": model_step,
        "data_steps": data_steps,
        "train_steps": usv_train_steps_singles + usv_train_steps_pos_prem + sv_train_steps,
        "eval_steps": usv_eval_steps_singles + usv_eval_pos_prem_steps + sv_eval_steps,
    };


{
    'steps_model_func': steps_model,
    'data_gen_steps_func': data_gen_steps,
    'sv_method_train_steps_func': sv_method_train_steps,
    'usv_method_train_steps_func': usv_method_train_steps,
    'create_method_prefix_func': create_method_prefix,
}
