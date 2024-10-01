local SPLIT = 'train';
local lcb_snli_prompts = import '../../belief-probing-repo/src/beliefprobing/bp_datasets/lcb_snli/templates.libsonnet';
local lcb_ent_bank_prompts = import '../../belief-probing-repo/src/beliefprobing/bp_datasets/lcb_ent_bank/templates.libsonnet';

{
    lcb_snli: {
        // original premises
        'snli-original_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: lcb_snli_prompts['pic_full_A'],
            split: SPLIT,
        },
        'snli-original_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: lcb_snli_prompts['pic_premise_negated_A'],
            split: SPLIT,
        },
        // shuffled premises
        'snli-shuffle_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_shuffle_premises',
            prompt: lcb_snli_prompts['pic_full_B'],
            split: SPLIT,
        },
        'snli-shuffle_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_shuffle_premises',
            prompt: lcb_snli_prompts['pic_premise_negated_B'],
            split: SPLIT,
        },
        // random characters instead of premises
        'snli-random_pos_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_random_bits',
            prompt: lcb_snli_prompts['pic_full_B'],
            split: SPLIT,
        },
        'snli-random_neg_prem': {
            name: 'lcb_snli',
            config: 'no_neutral_random_bits',
            prompt: lcb_snli_prompts['pic_premise_negated_B'],
            split: SPLIT,
        },
        // no premise
        'snli-no_prem': {
            name: 'lcb_snli',
            config: 'no_neutral',
            prompt: lcb_snli_prompts['pic_hypothesis_only'],
            split: SPLIT,
        },
    },
    lcb_ent_bank: {
        // original premises
        'entbank-original_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_base_all_true',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        'entbank-original_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_base_all_false',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        // shuffled premises
        'entbank-shuffle_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_distract_all_true',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        'entbank-shuffle_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_distract_all_false',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        // random characters instead of premises
        'entbank-random_pos_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_random_all_true',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        'entbank-random_neg_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_random_all_false',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
        // no premise
        'entbank-no_prem': {
            name: 'lcb_ent_bank',
            config: 'v3_none',
            prompt: lcb_ent_bank_prompts['question_full'],
            split: SPLIT,
        },
    },
}