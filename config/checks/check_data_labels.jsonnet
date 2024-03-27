local utils = import '../utils.libsonnet';
local common = import '../common.libsonnet';
local data = import '../data.libsonnet';
local models = import '../models.libsonnet';


local check_data_steps(dataset, model_key) =
    {
        ['tokenizer_' + model_key]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: models[model_key]['key']
        }
    } + {
        [model_key + '_' + data_key]: {
            "type": "load_data",
            dataset_name: data[dataset][data_key]['name'],
            split: data[dataset][data_key]['split'],
            tokenizer: {ref: 'tokenizer_' + model_key},
            dataset_config_name: data[dataset][data_key]['config'],
            prompt_name: data[dataset][data_key]['prompt'],
            model_type: 'decoder',
            add_period: true,
        }
        for data_key in std.objectFields(data[dataset])
    } + {
        ['check_' + dataset]: {
            type: 'check_consistent_data',
            all_data: [
                {ref: model_key + '_' + data_key}
                for data_key in std.objectFields(data[dataset])
            ],
        }
    };


local all_steps = [
    check_data_steps(dataset, model_key)
    for model_key in std.objectFields(models)
    for dataset in std.objectFields(data)
];


{
    'steps': utils.join_objects(all_steps)
}
