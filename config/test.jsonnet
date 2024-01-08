local SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

//local model = {
//    key: "roberta-large",
//    type: 'encoder',
//    layer: 24,
//    batch_size: 16,
//    revision: "main",
//};
//local model = {
//    key: "EleutherAI/gpt-j-6b",
//    type: 'decoder',
//    layer: 16,
//    batch_size: 16,
//    revision: "float16",
//};
local model = {
    key: "meta-llama/Llama-2-7b-hf",
    type: 'decoder',
    layer: 16,
    batch_size: 16,
    revision: "main",
};

local tokenizer = {pretrained_model_name_or_path: model['key']};
{
	"steps": {
	    "data": {
	        "type": "load_data",
            dataset_name: "lcb_snli",
            split: "validation",
            tokenizer: tokenizer,
            dataset_config_name: "no_neutral",
            prompt_name: "truth-full",
            model_type: model['type'],
	    },
	    "outputs": {
	        "type": "generate_hidden_states",
	        model: {
                "type": "transformers::AutoModelForCausalLM::from_pretrained",
                pretrained_model_name_or_path: model['key'],
                device_map: {"": "cuda:0"},
                torch_dtype: "float16",
                revision: model['revision'],
	        },
	        tokenizer: tokenizer,
	        dataset: {"ref": "data"},
	        batch_size: model['batch_size'],
            layer: model['layer'],
            model_type: model['type'],
	    },
	    'hidden_state_ranks': {
	        "type": "print_rank",
	        gen_out: {"ref": "outputs"},
	    },
	    'split_outputs': {
	        "type": "create_splits",
	        gen_out: {"ref": "outputs"},
	    },
	    'normalized_hidden_states': {
	        "type": "normalize",
	        data: {"ref": "split_outputs"},
	        var_normalize: false,
	    },
	    'var_normalized_hidden_states': {
	        "type": "normalize",
	        data: {"ref": "split_outputs"},
	        var_normalize: true,
	    },
	} + {
	    ['lm_head_baseline']: {
	        "type": "traineval_belief_probe",
            data: {"ref": "split_outputs"},
            probe: {"type": "lm_head_baseline"},
	    }
    } + {
	    ['logistic_baseline']: {
	        "type": "traineval_belief_probe",
            data: {"ref": "split_outputs"},
            probe: {"type": "lr_sklearn"},
	    }
	} + {
	    ['ccr']: {
	        "type": "traineval_belief_probe",
            data: {"ref": "var_normalized_hidden_states"},
            probe: {"type": "ccr", seed: 0},
	    }
	} + {
	    ['ccs_' + seed]: {
	        "type": "traineval_belief_probe",
            data: {"ref": "var_normalized_hidden_states"},
            probe: {"type": "ccs_gd", seed: seed},
	    }
	    for seed in SEEDS
	} + {
	    ['ccs_linear-prio=' + prio]: {
	        "type": "traineval_belief_probe",
            data: {"ref": "normalized_hidden_states"},
	        probe: {
                "type": "ccs_linear",
                priority: prio
            },
	    }
//	    for prio in [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 15]
	    for prio in [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
	} + {
	    ['massmean-iid=' + iid]: {
	        "type": "traineval_belief_probe",
            data: {"ref": "normalized_hidden_states"},
            probe: {
                "type": "mass_mean",
                iid: iid
            },
	    }
	    for iid in [false, true]
	}
}