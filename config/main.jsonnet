local MODELS = {
    'llama-7b': {
        key: "meta-llama/Llama-2-7b-hf",
        type: 'decoder',
        layer: 16,
        batch_size: 16,
        revision: "main",
    },
    'gpt-j': {
        key: "EleutherAI/gpt-j-6b",
        type: 'decoder',
        layer: 16,
        batch_size: 16,
        revision: "float16",
    },
    'roberta': {
        key: "roberta-large",
        type: 'encoder',
        layer: 24,
        batch_size: 16,
        revision: "main",
    }
};
local DATA = {
    // original premises
    'snli-original_pos_prem': {
        name: 'lcb_snli',
        config: 'no_neutral',
        prompt: 'pic-full',
    },
    'snli-original_neg_prem': {
        name: 'lcb_snli',
        config: 'no_neutral',
        prompt: 'pic-premise_negated',
    },
    // shuffled premises
    'snli-shuffle_pos_prem': {
        name: 'lcb_snli',
        config: 'no_neutral_shuffle_premises',
        prompt: 'pic-full',
    },
    'snli-shuffle_neg_prem': {
        name: 'lcb_snli',
        config: 'no_neutral_shuffle_premises',
        prompt: 'pic-premise_negated',
    },
    // random characters instead of premises
    'snli-random_pos_prem': {
        name: 'lcb_snli',
        config: 'no_neutral_random_bits',
        prompt: 'pic-full',
    },
    'snli-random_neg_prem': {
        name: 'lcb_snli',
        config: 'no_neutral_random_bits',
        prompt: 'pic-premise_negated',
    },
    // no premise
    'snli-no_prem': {
        name: 'lcb_snli',
        config: 'no_neutral',
        prompt: 'pic-hypothesis_only',
    },
};


local create_method_prefix(data_key, model_key, layer) =
    if layer == null then
        data_key + "|" + model_key + "|"
    else
        data_key + "|" + model_key + "|layer" + layer + "|";


local data_gen_steps(data_key, model_key, model_object, tokenizer_object, layers) =
    local data = DATA[data_key];
    local model = MODELS[model_key];

    // a prefix for the steps that involve this data and this model
    local prefix = create_method_prefix(data_key, model_key, null);

    {
	    [prefix + "data"]: {
	        "type": "load_data",
            dataset_name: data['name'],
            split: "validation",
            tokenizer: tokenizer_object,
            dataset_config_name: data['config'],
            prompt_name: data['prompt'],
            model_type: model['type'],
	    },
	    [prefix + "outputs"]: {
	        "type": "generate_hidden_states",
	        model: model_object,
	        tokenizer: tokenizer_object,
	        dataset: {"ref": prefix+"data"},
	        batch_size: model['batch_size'],
            all_layers: true,
            model_type: model['type'],
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
	    } for layer in layers
	} + {
	    [prefix + 'layer' + layer  + '|normalized_hidden_states']: {
	        "type": "normalize",
	        data: {"ref": prefix + 'layer' + layer + "|split_outputs"},
	        var_normalize: false,
	    } for layer in layers
	} + {
	    [prefix + 'layer' + layer + '|var_normalized_hidden_states']: {
	        "type": "normalize",
	        data: {"ref": prefix + 'layer' + layer + "|split_outputs"},
	        var_normalize: true,
	    } for layer in layers
	};


local CCS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
//local CCS_SEEDS = [0, 1];


local usv_method_train_steps(data_key, model_key, model_object, layer) =
    local prefix = create_method_prefix(data_key, model_key, layer);
    local calibration_prefix = create_method_prefix('snli-no_prem', model_key, layer);

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
//    } + ccs_trials + {
//        [prefix + 'train|ccs']: {
//            "type": "select_best",
//            train_data: {"ref": prefix+"normalized_hidden_states"},
//            calibration_data: {"ref": calibration_prefix+"normalized_hidden_states"},
//            probes: [
//                {"ref": prefix + 'trial|' + 'ccs_' + seed}
//                for seed in CCS_SEEDS
//            ],
//            probe_configs: std.objectKeysValues(ccs_trials)
//        }
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

local sv_method_train_steps(data_key, model_key, layer) =
    local prefix = create_method_prefix(data_key, model_key, layer);
    local calibration_prefix = create_method_prefix('snli-no_prem', model_key, layer);

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


// function that returns all steps for a given model
local steps_model(model_key, layers) =
    local model = MODELS[model_key];

    # load model
    local model_step = {
        "model": {
            type: "transformers::AutoModelForCausalLM::from_pretrained::step",
            pretrained_model_name_or_path: model['key'],
            device_map: {"": "cuda:0"},
            torch_dtype: "float16",
            revision: model['revision'],
        },
        "tokenizer": {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model['key'],
        }
    };

    local join_objects(objs) = std.foldl(function(acc, obj) acc + obj, objs, {});

    # load data and obtain hidden states
    local data_steps = join_objects([
        data_gen_steps(data_key, model_key, {"ref": "model"}, {"ref": "tokenizer"}, layers)
        for data_key in std.objectFields(DATA)
    ]);

    # train probing methods
    local usv_train_steps_singles = join_objects([
        usv_method_train_steps(data_key, model_key, {"ref": "model"}, layer)
        for layer in layers
        for data_key in std.objectFields(DATA) if data_key != 'snli-original_pos_prem'
    ]);
    local usv_train_steps_pos_prem = join_objects([
        usv_method_train_steps('snli-original_pos_prem', model_key, {"ref": "model"}, layer)
        for layer in layers
    ]);
    //local usv_train_steps_combined = usv_method_train_steps(...); TODO

    local sv_train_steps = join_objects([
        sv_method_train_steps('snli-original_pos_prem', model_key, layer)
        for layer in layers
    ]);

    # evaluate unsupervised methods on same data variant they were trained on
    local usv_eval_steps_singles = {
        [std.strReplace(probe_obj['key'], 'train', 'eval')]: {
            "type": "eval_belief_probe",
            data: probe_obj['value']['train_data'],
            probe: {"ref": probe_obj['key']},
        }
        for probe_obj in std.objectKeysValues(usv_train_steps_singles)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0
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
        for data_key in std.objectFields(DATA)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0
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
        for data_key in std.objectFields(DATA)
        for probe_obj in std.objectKeysValues(sv_train_steps)
        if std.length(std.findSubstr('train', probe_obj['key'])) > 0
    };

    {
        "other_steps": model_step + data_steps,
        "train_steps": usv_train_steps_singles + usv_train_steps_pos_prem + sv_train_steps,
        "eval_steps": usv_eval_steps_singles + usv_eval_pos_prem_steps + sv_eval_steps,
    };


{
    'steps_model_func': steps_model,
}