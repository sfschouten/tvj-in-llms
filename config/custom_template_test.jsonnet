local utils = import 'utils.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';

//local cities_prompts = import '../beliefprobing/datasets/got_cities/templates.libsonnet';
//local comparisons_prompts = import '../beliefprobing/datasets/got_comparisons/templates.libsonnet';
local sp_en_prompts = import '../beliefprobing/datasets/got_sp_en_trans/templates.libsonnet';

# dummy model, not used
local model_key = 'llama2-7b';
local model_config = models[model_key];

{
    "steps": {
        [model_key + "-tokenizer"]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model_config['key'],
        },
    } + {
        'data': {
            type: 'load_data',
//            dataset_name_or_path: 'got_cities',
//            dataset_name_or_path: 'got_comparisons',
            dataset_name_or_path: 'got_sp_en_trans',
//            dataset_config_name: 'add_negations',
            split: 'train',
            tokenizer: {ref: model_key + "-tokenizer"},
            model_type: model_config['type'],
            prompt_template:  sp_en_prompts['original'],
        }
    } + {
        ['extract_samples']: {
            type: 'extract_data_samples',
            dataset: {ref: 'data'}
        }
    }
}

//            prompt_template:  {
//                type: "custom_template",
//                name: "test123",
//                answer_choices: "correct ||| incorrect",
//                jinja:
//                    '{% for p,t in zip(premises, truths) %}'
//                  + 'The statement "{{ p }}" is {{ answer_choices[t] }}.\n'
//                  + '{% endfor %}'
//                  + '\n'
//                  + 'Answering "{{ question }}" with "{{ answer }}" is ||| {{ answer_choices[label] }}'
//            },
