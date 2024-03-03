{
    'llama-7b': {
        key: "meta-llama/Llama-2-7b-hf",
        type: 'decoder',
        layers: std.range(1, 32),
        batch_size: 16,
        revision: "main",
    },
//    'gpt-j': {
//        key: "EleutherAI/gpt-j-6b",
//        type: 'decoder',
//        layers: std.range(1, 28),
//        batch_size: 16,
//        revision: "float16",
//    },
//    'roberta': {
//        key: "roberta-large",
//        type: 'encoder',
//        layers: std.range(1, 24),
//        batch_size: 16,
//        revision: "main",
//    }
}