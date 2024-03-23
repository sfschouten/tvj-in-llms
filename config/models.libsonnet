{
    'llama2-7b': {
        key: "meta-llama/Llama-2-7b-hf",
        type: 'decoder',
        layers: std.range(1, 32),
        batch_size: 16,
        revision: "main",
        layer_template: 'model.layers.{}',
    },
    'llama2-13b': {
        key: "meta-llama/Llama-2-13b-hf",
        type: 'decoder',
        layers: std.range(1, 40),
        batch_size: 32,
        revision: "main",
        layer_template: 'model.layers.{}',
    },
    'gemma-7b': {
        key: "google/gemma-7b",
        type: 'decoder',
        layers: std.range(1, 28),
        batch_size: 12,
        revision: "main",
        layer_template: "model.layers.{}"
    },
    'gemma-7b-it': {
        key: "google/gemma-7b-it",
        type: 'decoder',
        layers: std.range(1, 28),
        batch_size: 12,
        revision: "main",
        layer_template: "model.layers.{}"
    },
//    'olmo-7b': {
//        key: 'allenai/OLMo-7B',
//        type: 'decoder',
//        layers: std.range(1,32),
//        batch_size: 16,
//        revision: 'main',
//        layer_template: 'model.transformer.blocks.{}',
//    },
//    'olmo-7b-instruct': {
//        key: 'allenai/OLMo-7B-Instruct',
//        type: 'decoder',
//        layers: std.range(1,32),
//        batch_size: 16,
//        revision: 'main',
//        layer_template: 'model.transformer.blocks.{}',
//    },
//    'pythia-2.8b': {
//        key: 'EleutherAI/pythia-2.8b',
//        type: 'decoder',
//        layers: std.range(1, 32),
//        batch_size: 64,
//        revision: 'main',
//        layer_template: 'gpt_neox.layers.{}',
//    },
//    'gpt-j': {
//        key: "EleutherAI/gpt-j-6b",
//        type: 'decoder',
//        layers: std.range(1, 28),
//        batch_size: 16,
//        revision: "float16",
//        layer_template: 'transformer.h.{}',
//    },
//    'roberta': {
//        key: "roberta-large",
//        type: 'encoder',
//        layers: std.range(1, 24),
//        batch_size: 16,
//        revision: "main",
//        layer_template: 'TODO.layers.{}',
//    }
}
