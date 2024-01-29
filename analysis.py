from functools import reduce
import operator
import hashlib

import matplotlib
import torch
import duckdb
import pandas
from duckdb import DuckDBPyConnection
from pandas import DataFrame
from tango import Step
from tango.format import TextFormat
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches


from integrations import DuckDBFormat
from evaluate import ProbeResults


PRIMITIVES = (bool, str, int, float)


@Step.register('duckdb_builder')
class DuckDBBuilder(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = '009e'

    def run(self, result_inputs: list, results: list[ProbeResults]) -> DuckDBPyConnection:
        dependencies_by_name = {step.name: step for step in self.dependencies}
        input_steps = [dependencies_by_name[_input.params['key']] for _input in result_inputs]

        def normalize(arraylike):
            if arraylike is None:
                return None
            return torch.tensor(arraylike).squeeze().float().tolist()
        results = [map(normalize, result) for result in results]

        def extract_relevant_info(step, prefix):
            return {
                f"{prefix}_{type(step).__name__}_{key}": value for key, value in step.kwargs.items()
                if isinstance(value, PRIMITIVES)
            }

        def merge_dicts(dicts):
            return reduce(operator.or_, dicts, {})

        info = [
            merge_dicts([
                extract_relevant_info(dependency, 'eval')
                for dependency in step.kwargs['data'].recursive_dependencies
            ] + [
                extract_relevant_info(dependency, 'train')
                for dependency in step.kwargs['probe'].recursive_dependencies
            ] + [
                {f'PROBE_{key}': value for key, value in step.kwargs['probe'].config['probe'].items()}
                | {'STEP_NAME': step.name}
            ])
            for step in input_steps
        ]

        for info_dict in info:
            keys = set(info_dict.keys()) - {
                # 'train_LoadData_prompt_name', 'train_LoadData_dataset_config_name',  # TODO add back and deal with
                'eval_LoadData_prompt_name', 'eval_LoadData_dataset_config_name',
                'STEP_NAME'
            }
            values = [str(info_dict[key]) for key in sorted(keys)]
            info_dict['data_group_id'] = hashlib.sha256("".join(values).encode('utf8')).hexdigest()

        df = pandas.DataFrame(info)
        labels, predictions, directions = zip(*results)
        df['label'] = labels
        df['direction'] = directions
        df['prediction'] = predictions

        # store in duckdb
        db = duckdb.connect()
        db.sql('CREATE TABLE results AS SELECT * FROM df')

        return db


@Step.register('direction_similarity')
class DirectionSimilarity(Step[str]):
    FORMAT = TextFormat

    def run(self, db: DuckDBPyConnection, **kwargs) -> str:
        # calculate cosine similarity between pairs of directions

        similarities = db.sql(
            "SELECT r1.STEP_NAME, r2.STEP_NAME, list_cosine_similarity(r1.direction, r2.direction)\n"
            "FROM results AS r1\n"
            "CROSS JOIN results AS r2\n"
        ).df()

        csv_string = similarities.to_csv()
        return csv_string


@Step.register('context_sensitivity')
class ContextSensitivity(Step[str]):
    FORMAT = TextFormat
    VERSION = "001a"

    def run(self, db: DuckDBPyConnection, **kwargs) -> str:
        # compare evaluations on positive premises to all others

        # join results on itself
        df: DataFrame = db.sql(
            "SELECT r1.data_group_id, r1.PROBE_type, "
            "r1.STEP_NAME AS step_name1, r2.STEP_NAME AS step_name2, "
            "r1.train_LoadData_prompt_name AS train_prompt1, r1.train_LoadData_dataset_config_name AS train_config1, "
            "r2.train_LoadData_prompt_name AS train_prompt2, r2.train_LoadData_dataset_config_name AS train_config2, "
            "r1.eval_LoadData_prompt_name AS eval_prompt1, r1.eval_LoadData_dataset_config_name AS eval_config1, "
            "r2.eval_LoadData_prompt_name AS eval_prompt2, r2.eval_LoadData_dataset_config_name AS eval_config2, "
            "r1.train_AutoModelLoaderPretrained_pretrained_model_name_or_path AS model_name, "
            "r1.train_Generate_layer AS model_layer, "
            "r1.train_Generate_token_idx AS token_idx, "
            "r1.prediction AS prediction1, r2.prediction AS prediction2, r1.label AS label\n"
            "FROM results AS r1\n"
            "INNER JOIN results AS r2\n"
            "ON r1.data_group_id = r2.data_group_id "
            "AND r1.eval_LoadData_prompt_name = 'truth-full' "
            "AND r1.eval_LoadData_dataset_config_name = 'no_neutral' \n"
            "AND (r2.eval_LoadData_prompt_name != 'truth-full' "
            "OR r2.eval_LoadData_dataset_config_name != 'no_neutral')\n"
        ).df()
        difference = np.array(df['prediction1'].values.tolist()) - np.array(df['prediction2'].values.tolist())
        df['difference'] = list(difference)
        df['premise_asserted'] = df['eval_prompt2'] != 'truth-premise_negated'
        df['premise_origin'] = df['eval_config2'].replace({
            'no_neutral_shuffle_premises': 'shuffled',
            'no_neutral_random_bits': 'random',
            'no_neutral': 'original'
        })

        # calculate metrics
        pth = 20
        pth_key = f"{pth}pth"
        df[pth_key] = np.percentile(difference, axis=1, q=pth)

        COLUMNS = {"data_group_id", "PROBE_type", "model_name", "model_layer", "token_idx"}
        COL_STR = ", ".join(COLUMNS)
        df2 = db.sql(
            f'SELECT list("{pth_key}"), list(step_name2), {COL_STR}'
            # f'list(eval_prompt1), list(eval_config1), list(eval_prompt2), list(eval_config2),'
            # f'list(train_prompt1), list(train_config1), list(train_prompt2), list(train_config2), '
            f'\nFROM df GROUP BY {COL_STR}\n'
        ).df()

        # GENERATE PLOTS
        g = sns.FacetGrid(
            df.explode(['label', 'difference']),
            row='step_name1', col='label', margin_titles=False, ylim=(-1, 1), aspect=2
        )

        # add a horizontal line
        for ax in g.axes.flatten():
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        # Draw the plots
        g.map(sns.boxplot, 'premise_origin', 'difference', 'premise_asserted', showfliers=False, palette='muted')
        g.map(sns.stripplot, 'premise_origin', 'difference', 'premise_asserted', size=3, alpha=0.2, palette='muted',
              jitter=.2, linewidth=0.1, dodge=True)

        # Create custom legend elements
        legend_elements = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(["False", "True"], sns.color_palette("muted"))
        ] + [mpatches.Patch(color='grey', label='N/A')]
        g.add_legend(handles=legend_elements)

        for ax in g.axes.flat:
            # Change boxplot color
            box_children = [child for child in ax.get_children() if isinstance(child, matplotlib.patches.PathPatch)]
            if box_children:
                box_children[-1].set_facecolor('grey')  # Replace 'your_box_color' with the desired color

            # Change stripplot color
            stripplot_children = [child for child in ax.get_children() if
                                  isinstance(child, matplotlib.collections.PathCollection)]
            if stripplot_children:
                stripplot_children[-1].set_facecolor('grey')
                stripplot_children[-1].set_edgecolor('grey')

            title = ax.get_title().replace(' | ', '\n| ')
            ax.set_title(title, fontsize=9)

        g.fig.tight_layout()
        g.fig.savefig(self.work_dir.parent / 'plot.pdf', format='pdf', dpi=300)

        return None
