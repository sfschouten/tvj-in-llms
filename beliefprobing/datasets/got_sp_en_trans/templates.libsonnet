{
    'original': {
        type: 'custom_template',
        name: 'original',
        answer_choices: ' ||| ',
        jinja: '{{ start }} ||| {{ [neg_end, pos_end][label] }}'
    }
}