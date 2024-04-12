 {
    pic_full: {
        type: 'custom_template',
        name: 'pic_full',
        answer_choices: 'incorrect ||| correct',
        jinja: 'You are looking at a picture.\n'
             + 'Describing it as "{{ premise }}" is correct.\n'
             + 'Saying (about the picture) that: "{{ hypothesis }}" is ||| {{ answer_choices[label] }}.',
    },
    pic_premise_negated: {
        type: 'custom_template',
        name: 'pic_premise_negated',
        answer_choices: 'incorrect ||| correct',
        jinja: 'You are looking at a picture.\n'
             + 'Describing it as "{{ premise }}" is incorrect.\n'
             + 'Saying (about the picture) that: "{{ hypothesis }}" is ||| {{ answer_choices[label] }}.',
    },
    pic_hypothesis_only: {
        type: 'custom_template',
        name: 'pic_hypothesis_only',
        answer_choices: 'incorrect ||| correct',
        jinja: 'You are looking at a picture.\n'
             + 'Saying (about the picture) that: "{{ hypothesis }}" is ||| {{ answer_choices[label] }}.',
    },
}
