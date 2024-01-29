local model = {
    key: "roberta-large",
    type: 'encoder',
    layer: 24,
    batch_size: 16,
    revision: "main",
};

local tokenizer = {pretrained_model_name_or_path: model['key']};

{
	"steps": {
	    "data": {
	        "type": "load_data",
            dataset_name: "lcb_ent_bank",
            split: "validation",
            tokenizer: tokenizer,
            prompt_i: 0,
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
    }
}