local models = import 'models.libsonnet';

{
    "steps": {
        ['tokenizer' + model['key']]: {
            type: "transformers::AutoTokenizer::from_pretrained::step",
            pretrained_model_name_or_path: model['value']['key']
        }
        for model in std.objectKeysValues(models)
    } + {
        ['tokens_' + model_key]: {
            type: 'check_tokenization',
            tokenizer: {ref: 'tokenizer' + model_key},
        }
        for model_key in std.objectFields(models)
    }
}