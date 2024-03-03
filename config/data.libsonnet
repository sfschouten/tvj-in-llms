local SPLIT = 'train';
{
    snli: {
        // original premises
        'snli-original_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: 'pic-full',
            split: SPLIT,
        },
        'snli-original_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: 'pic-premise_negated',
            split: SPLIT,
        },
        // shuffled premises
        'snli-shuffle_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_shuffle_premises',
            prompt: 'pic-full',
            split: SPLIT,
        },
        'snli-shuffle_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_shuffle_premises',
            prompt: 'pic-premise_negated',
            split: SPLIT,
        },
        // random characters instead of premises
        'snli-random_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_random_bits',
            prompt: 'pic-full',
            split: SPLIT,
        },
        'snli-random_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_random_bits',
            prompt: 'pic-premise_negated',
            split: SPLIT,
        },
        // no premise
        'snli-no_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: 'pic-hypothesis_only',
            split: SPLIT,
        },
    },
    entbank: {
        // original premises
        'entbank-original_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_base_all_true',
            prompt: 'truth',
            split: SPLIT,
        },
        'entbank-original_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_base_all_false',
            prompt: 'truth',
            split: SPLIT,
        },
        // shuffled premises
        'entbank-shuffle_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_distract_all_true',
            prompt: 'truth',
            split: SPLIT,
        },
        'entbank-shuffle_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_distract_all_false',
            prompt: 'truth',
            split: SPLIT,
        },
        // random characters instead of premises
        'entbank-random_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_random_all_true',
            prompt: 'truth',
            split: SPLIT,
        },
        'entbank-random_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_random_all_false',
            prompt: 'truth',
            split: SPLIT,
        },
        // no premise
        'entbank-no_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_none',
            prompt: 'truth',
            split: SPLIT,
        },
    },
}