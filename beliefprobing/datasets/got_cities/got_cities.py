import pandas as pd

import datasets
from datasets import DownloadManager, load_dataset, DatasetInfo


_COUNTRIES = {
    'Kenya': 'Kenya',
    'Croatia': 'Croatia',
    'Austria': 'Austria',
    'Peru': 'Peru',
    'Zambia': 'Zambia',
    'Tajikistan': 'Tajikistan',
    'Niger': 'Niger',
    'Congo, Democratic Republic of the': 'the Democratic Republic of the Congo',
    'Algeria': 'Algeria',
    'Trinidad and Tobago': 'Trinidad and Tobago',
    'Cyprus': 'Cyprus',
    'Mauritania': 'Mauritania',
    'Uruguay': 'Uruguay',
    'Slovenia': 'Slovenia',
    'Saint Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Bolivia': 'Bolivia',
    'Malawi': 'Malawi',
    'Bangladesh': 'Bangladesh',
    'Turkey': 'Turkey',
    'Vanuatu': 'Vanuatu',
    'Madagascar': 'Madagascar',
    'Hungary': 'Hungary',
    'Kyrgyzstan': 'Kyrgyzstan',
    'New Zealand': 'New Zealand',
    'Uzbekistan': 'Uzbekistan',
    'Iran, Islamic Rep. of': 'Iran',
    'Togo': 'Togo',
    'Tonga': 'Tonga',
    'Monaco': 'Monaco',
    'Luxembourg': 'Luxembourg',
    'Costa Rica': 'Costa Rica',
    'Belize': 'Belize',
    'Montenegro': 'Montenegro',
    'Spain': 'Spain',
    'Lebanon': 'Lebanon',
    'Poland': 'Poland',
    'Tanzania, United Republic of': 'Tanzania',
    'Australia': 'Australia',
    'Angola': 'Angola',
    'Saint Kitts and Nevis': 'Saint Kitts and Nevis',
    'Panama': 'Panama',
    'Samoa': 'Samoa',
    'Switzerland': 'Switzerland',
    'Burundi': 'Burundi',
    'Mozambique': 'Mozambique',
    'Papua New Guinea': 'Papua New Guinea',
    'Bulgaria': 'Bulgaria',
    'Chile': 'Chile',
    'Mali': 'Mali',
    'Venezuela, Bolivarian Rep. of': 'Venezuela',
    'South Africa': 'South Africa',
    'United Kingdom': 'the United Kingdom',
    'Comoros': 'Comoros',
    'Japan': 'Japan',
    'Albania': 'Albania',
    'Senegal': 'Senegal',
    'Guatemala': 'Guatemala',
    'Guinea': 'Guinea',
    'Malaysia': 'Malaysia',
    'Yemen': 'Yemen',
    'Nauru': 'Nauru',
    'Syrian Arab Republic': 'Syria',
    'Slovakia': 'Slovakia',
    'Germany': 'Germany',
    'Ecuador': 'Ecuador',
    'Lithuania': 'Lithuania',
    'Dominica': 'Dominica',
    'Azerbaijan': 'Azerbaijan',
    'Sudan, The Republic of': 'Sudan',
    'Seychelles': 'Seychelles',
    'Kiribati': 'Kiribati',
    'Iraq': 'Iraq',
    'Namibia': 'Namibia',
    'Congo': 'the Republic of the Congo',
    'Andorra': 'Andorra',
    'Canada': 'Canada',
    'Korea, Republic of': 'South Korea',
    'Bahamas': 'the Bahamas',
    'Sierra Leone': 'Sierra Leone',
    'Brazil': 'Brazil',
    'Finland': 'Finland',
    'Ukraine': 'Ukraine',
    'Norway': 'Norway',
    'Russian Federation': 'Russia',
    'Cuba': 'Cuba',
    'Sao Tome and Principe': 'Sao Tome and Principe',
    'Estonia': 'Estonia',
    'Portugal': 'Portugal',
    'Greece': 'Greece',
    'Bhutan': 'Bhutan',
    'Latvia': 'Latvia',
    'Central African Republic': 'the Central African Republic',
    'Zimbabwe': 'Zimbabwe',
    'Lesotho': 'Lesotho',
    'Moldova, Republic of': 'Moldova',
    'Mauritius': 'Mauritius',
    'Palau': 'Palau',
    'Nicaragua': 'Nicaragua',
    'Djibouti': 'Djibouti',
    'Gabon': 'Gabon',
    'Dominican Republic': 'the Dominican Republic',
    'Qatar': 'Qatar',
    'Bosnia and Herzegovina': 'Bosnia and Herzegovina',
    'Kazakhstan': 'Kazakhstan',
    'Maldives': 'the Maldives',
    'China': 'China',
    'Viet Nam': 'Vietnam',
    "Korea, Dem. People's Rep. of": 'North Korea',
    'Myanmar': 'Myanmar',
    'Turkmenistan': 'Turkmenistan',
    'Barbados': 'Barbados',
    'San Marino': 'San Marino',
    'Romania': 'Romania',
    'Armenia': 'Armenia',
    'United Arab Emirates': 'the United Arab Emirates',
    'Malta': 'Malta',
    'Uganda': 'Uganda',
    'United States': 'the United States',
    'Saudi Arabia': 'Saudi Arabia',
    'Ethiopia': 'Ethiopia',
    'Guyana': 'Guyana',
    'Benin': 'Benin',
    'India': 'India',
    'Macedonia, The former Yugoslav Rep. of': 'Macedonia',
    'Philippines': 'the Philippines',
    'Mexico': 'Mexico',
    'Fiji': 'Fiji',
    'Bahrain': 'Bahrain',
    'Belarus': 'Belarus',
    'Afghanistan': 'Afghanistan',
    "Côte d'Ivoire": "Côte d'Ivoire",
    'France': 'France',
    'Kuwait': 'Kuwait',
    'Czech Republic': 'the Czech Republic',
    'Egypt': 'Egypt',
    'Jordan': 'Jordan',
    'Gambia': 'the Gambia',
    'Equatorial Guinea': 'Equatorial Guinea',
    'Oman': 'Oman',
    'Denmark': 'Denmark',
    'Haiti': 'Haiti',
    'El Salvador': 'El Salvador',
    'Liberia': 'Liberia',
    'Tuvalu': 'Tuvalu',
    'Burkina Faso': 'Burkina Faso',
    'Chad': 'Chad',
    'Guinea-Bissau': 'Guinea-Bissau',
    'Cape Verde': 'Cape Verde',
    'Somalia': 'Somalia',
    'Indonesia': 'Indonesia',
    'Tunisia': 'Tunisia',
    'Belgium': 'Belgium',
    'Liechtenstein': 'Liechtenstein',
    'Colombia': 'Colombia',
    "Lao People's Dem. Rep.": 'Laos',
    'Timor-Leste': 'Timor-Leste',
    'Honduras': 'Honduras',
    'Italy': 'Italy',
    'Serbia': 'Serbia',
    'Netherlands': 'the Netherlands',
    'Jamaica': 'Jamaica',
    'Eritrea': 'Eritrea',
    'Nepal': 'Nepal',
    'Swaziland': 'Swaziland',
    'Antigua and Barbuda': 'Antigua and Barbuda',
    'Rwanda': 'Rwanda',
    'Paraguay': 'Paraguay',
    'Sri Lanka': 'Sri Lanka',
    'Iceland': 'Iceland',
    'Morocco': 'Morocco',
    'Suriname': 'Suriname',
    'Argentina': 'Argentina',
    'Mongolia': 'Mongolia',
    'Botswana': 'Botswana',
    'Thailand': 'Thailand',
    'Cameroon': 'Cameroon',
    'Ireland': 'Ireland',
    'Nigeria': 'Nigeria',
    'Cambodia': 'Cambodia',
    'Sweden': 'Sweden',
    'Pakistan': 'Pakistan',
    'Ghana': 'Ghana',
    'Singapore': 'Singapore',
}

_DESCRIPTION = """
Cities dataset from "The Geometry of Truth: Emergent linear structure in Large Language Model representations of 
True/False datasets" by Marks and Tegmark.
"""

_URLS = {
    "geo_names": "https://github.com/saprmarks/geometry-of-truth/raw/main/datasets/geonames.csv"
}


class CitiesConfig(datasets.BuilderConfig):

    def __init__(self, add_negated_samples: bool, add_false_country_samples: bool, **kwargs):
        super().__init__(**kwargs)
        self.add_negated_samples = add_negated_samples
        self.add_false_country_samples = add_false_country_samples


class Cities(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        CitiesConfig(name="original", add_negated_samples=False, add_false_country_samples=False),
        CitiesConfig(name="add_incorrect_countries", add_negated_samples=False, add_false_country_samples=True),
        CitiesConfig(name="add_negations", add_negated_samples=True, add_false_country_samples=False),
        CitiesConfig(name="add_both", add_negated_samples=True, add_false_country_samples=True),
    ]

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'label': datasets.Value('int32'),
                'polarity': datasets.Value('int32'),
                'city': datasets.Value('string'),
                'country_i': datasets.Value('int32'),
                'countries': datasets.Sequence(datasets.Value('string')),
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        geo_names_csv_path = dl_manager.download(_URLS['geo_names'])
        df = pd.read_csv(geo_names_csv_path)

        def is_valid(row):
            name = row['ASCII Name']
            population = row['Population']
            country = row['Country name EN']

            # check that the country is valid
            if country not in _COUNTRIES:
                return False

            # check population is larger than 500000
            if not population > 500000:
                return False

            # check that not a city-state
            if name in country:
                return False

            # check there is no other city with the same name
            if len(df[df['ASCII Name'] == name]) > 1:
                return False

            return True

        # filter valid rows
        df = df[df.apply(is_valid, axis=1)]

        train_df = df.iloc[:len(df)//2]
        test_df = df.iloc[len(df)//2:]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"df": train_df},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"df": test_df},
            ),
        ]

    def _generate_examples(self, df):
        for idx, row in df.iterrows():
            city = row['ASCII Name']

            true_country = _COUNTRIES[row['Country name EN']]
            false_country = true_country
            while false_country == true_country:
                false_country = _COUNTRIES[df['Country name EN'].sample(1).iloc[0]]

            countries = [(1, true_country)]
            if self.config.add_false_country_samples:
                countries.append((0, false_country))

            polarities = [False, True] if self.config.add_negated_samples else [True]

            variants = [
                {
                    'label': bool(country_i) == polarity,
                    'polarity': polarity,
                    'city': city,
                    'country_i': country_i,
                    'countries': [false_country, true_country],
                }
                for country_i, country in countries
                for polarity in polarities
            ]

            for j, variant in enumerate(variants):
                yield idx * (len(countries) + len(polarities)) + j, variant


if __name__ == "__main__":
    from pprint import pprint
    datasets.disable_caching()

    for config in [""]:

        print(' ####  ' + config.upper() + '  #### ')

        # load a dataset
        dataset = load_dataset(__file__, name=config).shuffle()

        # print some samples
        for i, test in enumerate(dataset['train']):
            print(i)
            pprint(test)
            print()
            if i >= 9:
                break
