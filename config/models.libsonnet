{
    'llama2-7b': {
        key: "meta-llama/Llama-2-7b-hf",
        type: 'decoder',
        layers: std.range(1, 32),
        batch_size: 16,
        revision: "main",
        layer_template: 'model.layers.{}',
    },
//    'gemma-7b': {
//        key: "google/gemma-7b",
//        type: 'decoder',
//        layers: std.range(1, 28),
//        batch_size: 12,
//        revision: "main",
//    },
    'llama2-13b': {
        key: "meta-llama/Llama-2-13b-hf",
        type: 'decoder',
        layers: std.range(1, 40),
        batch_size: 32,
        revision: "main",
    },
    'gpt-j': {
        key: "EleutherAI/gpt-j-6b",
        type: 'decoder',
        layers: std.range(1, 28),
        batch_size: 16,
        revision: "float16",
    },
//    'roberta': {
//        key: "roberta-large",
//        type: 'encoder',
//        layers: std.range(1, 24),
//        batch_size: 16,
//        revision: "main",
//    }
}
