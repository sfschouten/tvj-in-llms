local dataset = std.extVar('dataset');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';


local data_key = dataset + '-original_pos_prem';
local data_config = data[dataset][data_key];


local data_and_train_steps(model_key, model_config) =
    local train_steps = utils.join_objects([
        usv_method_train_steps(data_key, model_key, layer)
        for layer in model_config['layers']
    ]) + utils.join_objects([
        sv_method_train_steps(data_key, model_key, layer)
        for layer in model_config['layers']
    ]);

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
        } + data_gen_steps(
            data_key, data_config, model_key, model_config,
            {"ref": model_key}, {"ref": model_key + "-tokenizer"}
        ) + train_steps + {
            "intervened_outputs": {
                type: "generate_with_intervention",
                model: {ref: model_key},
                tokenizer: {ref: model_key + "-tokenizer"},
                dataset: {"ref": common.create_method_prefix_func()+"data"},
                batch_size: model_config['batch_size'],
                all_layers: true,
                model_type: model_config['type'],
            }
//            for train_step in train_steps
        },
    };


{
    "steps": {
	    [prefix + "data"]: {
	        "type": "load_data",
            dataset_name: data_config['name'],
            split: data_config['split'],
            tokenizer: tokenizer_object,
            dataset_config_name: data_config['config'],
            prompt_name: data_config['prompt'],
            model_type: model_config['type'],
	    },
        "intervened_outputs": {
            type: "generate_with_intervention",
	        model: model_object,
	        tokenizer: tokenizer_object,
	        dataset: {"ref": prefix+"data"},
	        batch_size: model_config['batch_size'],
            all_layers: true,
            model_type: model_config['type'],
        },
//        {
//            [prefix + 'layer' + layer + '|split_outputs']: {
//                "type": "create_splits",
//                gen_out: {"ref": prefix+"outputs"},
//                layer_index: layer
//            } for layer in model_config['layers']
//        } + {
//            [prefix + 'layer' + layer  + '|normalized_hidden_states']: {
//                "type": "normalize",
//                data: {"ref": prefix + 'layer' + layer + "|split_outputs"},
//                var_normalize: false,
//            } for layer in model_config['layers']
//        } + {
//            [std.strReplace(probe_obj['key'], 'train', 'eval@' + data_key)]: {
//                "type": "eval_belief_probe",
//                data: {
//                    "ref": create_method_prefix(data_key, model_key, null)
//                       + std.join('|', std.split(probe_obj['value']['train_data']['ref'], '|')[2:4])
//                },
//                probe: {"ref": probe_obj['key']},
//            }
//            for probe_obj in std.objectKeysValues(train_steps)
//            for data_key in std.objectFields(dataset_config)
//            if std.length(std.findSubstr('train', probe_obj['key'])) > 0    # skip individual trials
//            if
//        }
    }
}
