{
    original: {
        type: 'custom_template',
        name: 'original',
        answer_choices: ' ||| ',
        # [label=0 & diff<0  |  label=1 & diff>0]: 'larger'
        # [label=0 & diff>0  |  label=1 & diff<0]: 'smaller'
        jinja: '{{ n1_text }} is  ||| {{ ["smaller", "larger"][label==(diff>0)] }} than {{ n2_text }}.',
    }
}