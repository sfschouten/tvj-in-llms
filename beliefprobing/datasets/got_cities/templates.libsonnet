{
    original: {  # varies the country to set the truth of the statement
        type: "custom_template",
        name: "original",
        answer_choices: " ||| ",
        jinja: 'The city of {{ city }} is {% if not polarity %}not {% endif %}in ||| {{ countries[label] }}.'
    },
    polarity: {  # varies the polarity to set the truth of the statement
        type: "custom_template",
        name: "polarity",
        answer_choices: " ||| ",
        jinja: 'The city of {{ city }} is  ||| {% if label != country_i %}not {% endif %}in {{ countries[country_i] }}.'
    },
    meta_polarity: {  # varies a meta-level judgment about the sentence to set the truth of the statement
        type: "custom_template",
        name: "meta_polarity",
        answer_choices: "incorrect ||| correct",
        jinja: 'Saying that '
             + '"The city of {{ city }} is {% if not polarity %}not {% endif %}in {{ countries[country_i] }}."'
             + ' is ||| {{ answer_choices[label] }}.'
    }
}