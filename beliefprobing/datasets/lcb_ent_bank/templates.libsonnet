{
    statements: {
        type: 'custom_template',
        name: 'statements',
        answer_choices: 'incorrect ||| correct',
        jinja: '{% for p,t in zip(premises, truths) %}'
             + 'The statement "{{ p }}" is {{ answer_choices[t] }}.\n'
             + '{% endfor %}'
             + '\n'
             + 'Answering "{{ question }}" with "{{ answer }}" is ||| {{ answer_choices[label] }}.',
    },
    question_short: {
        type: 'custom_template',
        name: 'question_short',
        answer_choices: 'incorrect ||| correct',
        jinja: 'Question: "{{ question }}"\n'
             + '{% for p,t in zip(premises, truths) %}'
             + 'The statement "{{ p }}" is {{ answer_choices[t] }}.\n'
             + '{% endfor %}'
             + '\n'
             + 'Answering the question with "{{ answer }}" is ||| {{ answer_choices[label] }}.',
    },
    question_full: {
        type: 'custom_template',
        name: 'question_full',
        answer_choices: 'incorrect ||| correct',
        jinja: 'You are given the following question:\n'
             + '> {{ question }}\n'
             + '{% for p,t in zip(premises, truths) %}'
             + 'The statement "{{ p }}" is {{ answer_choices[t] }}.\n'
             + '{% endfor %}'
             + '\n'
             + 'Answering the question with "{{ answer }}" is ||| {{ answer_choices[label] }}.',
    },
}