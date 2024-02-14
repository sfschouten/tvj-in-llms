from functools import reduce
import operator
import hashlib

import matplotlib
import pandas as pd
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
from scipy.stats import hmean

from integrations import DuckDBFormat
from evaluate import ProbeResults

PRIMITIVES = (bool, str, int, float)


@Step.register('duckdb_builder')
class DuckDBBuilder(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = '014'

    def run(self, result_inputs: list, results: list[ProbeResults]) -> DuckDBPyConnection:
        dependencies_by_name = {step.name: step for step in self.dependencies}
        input_steps = [dependencies_by_name[_input.params['key']] for _input in result_inputs]

        def normalize(arraylike):
            if arraylike is None:
                return None
            return torch.tensor(arraylike).squeeze().float().tolist()

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
                {f'PROBE_{key}': value for key, value in result[0].config.items()}
                | {'EVAL_STEP_NAME': step.name}
                | {
                    f'TRAIN_STEP_NAME.{i}': part
                    for i, part in enumerate(result[0].train_step_name.split('|'))
                }
            ])
            for result, step in zip(results, input_steps)
        ]

        # create groups of results to analyse together.
        for info_dict in info:
            keys = set(info_dict.keys()) - {'EVAL_STEP_NAME'}
            probe_details = {key for key in info_dict.keys() if key.startswith('PROBE')}
            # results that only differ in the data variant used for training/evaluation belong to the same level 0 group
            excl = [{
                'train_LoadData_prompt_name', 'train_LoadData_dataset_config_name', 'TRAIN_STEP_NAME.1',
                'eval_LoadData_prompt_name', 'eval_LoadData_dataset_config_name',
            } | probe_details]
            # level 1 groups consist of level 0 groups that also differ in probe type
            excl.append(excl[0] | {'TRAIN_STEP_NAME.3', 'train_Normalize_var_normalize', 'eval_CreateSplits_layer_index'})
            for lvl, lvl_excl in enumerate(excl):
                values = [str(info_dict[key]) for key in sorted(keys - lvl_excl)]
                info_dict[f'data_group_l{lvl}_id'] = hashlib.sha256("".join(values).encode('utf8')).hexdigest()

        df = pandas.DataFrame(info)
        _, labels, predictions, directions = zip(*results)
        df['label'] = list(map(normalize, labels))
        df['direction'] = list(map(normalize, directions))
        df['prediction'] = list(map(normalize, predictions))

        # store in duckdb
        db = duckdb.connect()
        db.sql('CREATE TABLE results AS SELECT * FROM df')

        return db


@Step.register('context_sensitivity')
class ContextSensitivity(Step[str]):
    FORMAT = TextFormat
    VERSION = "002"

    def run(self, db: DuckDBPyConnection, **kwargs) -> str:
        df: DataFrame = db.sql("SELECT * FROM results").df()

        # convert to ndarray
        df['prediction'] = df.apply(lambda x: np.array(x.prediction), axis=1)
        df['direction'] = df.apply(lambda x: np.array(x.direction), axis=1)
        df['label'] = df.apply(lambda x: np.array(x.label), axis=1)

        # nicer names
        df = df.rename(columns={
            'train_LoadData_prompt_name': 'train_prompt',
            'train_LoadData_dataset_config_name': 'train_config',
            'eval_LoadData_prompt_name': 'eval_prompt',
            'eval_LoadData_dataset_config_name': 'eval_config',
        })
        df['premise_asserted'] = df['eval_prompt'] != 'truth-premise_negated'
        df['premise_origin'] = df['eval_config'].replace({
            'no_neutral_shuffle_premises': 'shuffled',
            'no_neutral_random_bits': 'random',
            'no_neutral': 'original'
        })
        df.loc[df['eval_prompt'] == 'truth-hypothesis_only', 'premise_origin'] = 'no-premise'
        df['same_variant'] = (df['eval_prompt'] == df['train_prompt']) & (df['eval_config'] == df['train_config'])

        pos_origin_idx = (df['premise_origin'] == 'original') & (df['eval_prompt'] == 'truth-full')
        df['same_variant_grp'] = df['same_variant'] | pos_origin_idx
        df['pos_origin_variant_grp'] = ~df['same_variant'] | pos_origin_idx

        err_cols = ['error_1', 'error_2', 'error_3ab', 'error_3cd', 'error_4']
        total_err_cols = [f'total_{err}' for err in err_cols]
        df = df.assign(**{err_col: None for err_col in err_cols + total_err_cols})

        subgroup_aggr_stats = {}

        for l1_name, l1_df in df.groupby(by='data_group_l1_id'):
            for l0_name, l0_df in l1_df.groupby(by='data_group_l0_id'):
                original_pos = l0_df[(l0_df['premise_origin'] == 'original') & (l0_df['eval_prompt'] == 'truth-full')]

                def process_subgroup(sgr_df, sg_name):
                    a, o = sgr_df['premise_asserted'], sgr_df['premise_origin']
                    h_nly_i = sgr_df[sgr_df['eval_prompt'] == 'truth-hypothesis_only'].index
                    hyp_only_pred_vals = sgr_df.loc[h_nly_i].prediction.values[0]

                    for origin, error_i in [('random', 1), ('shuffled', 2)]:
                        for asserted in [True, False]:
                            # error types 1 & 2
                            bl_i = sgr_df[(o == origin) & (a == asserted)].index
                            assert len(bl_i) == 1
                            err = abs(sgr_df.loc[bl_i].prediction.values[0] - hyp_only_pred_vals)
                            df.at[bl_i[0], f'error_{error_i}'] = err

                    labels = original_pos.label.values[0].astype(bool)

                    # error type 3
                    o_neg_i = sgr_df[(o == 'original') & ~a].index
                    assert len(o_neg_i) == 1
                    o_neg_vals = sgr_df.loc[o_neg_i].prediction.values[0]
                    o_pos_vals = original_pos.prediction.values[0]

                    err_3a = labels * (o_neg_vals - hyp_only_pred_vals).clip(min=0)
                    err_3b = ~labels * (-o_neg_vals + hyp_only_pred_vals).clip(min=0)
                    err_3c = labels * (-o_pos_vals + hyp_only_pred_vals).clip(min=0)
                    err_3d = ~labels * (o_pos_vals - hyp_only_pred_vals).clip(min=0)
                    df.at[o_neg_i[0], 'error_3ab'] = err_3a + err_3b
                    df.at[h_nly_i[0], 'error_3cd'] = err_3c + err_3d

                    # error type 4
                    df.at[o_neg_i[0], 'error_4'] = abs(o_neg_vals - o_pos_vals)

                    # calculate totals
                    for col in df.columns:
                        if col.startswith('error_'):
                            df.loc[sgr_df.index, 'total_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else x.sum()
                            )

                    # aggregate
                    aggregates = df.loc[sgr_df.index, total_err_cols].mean(axis=0).to_dict()
                    total_error_3 = (aggregates.pop('total_error_3ab') + aggregates.pop('total_error_3cd')) / 2
                    aggregates |= {'total_error_3': total_error_3}
                    aggregates |= {
                        'total_error_A': hmean([
                            aggregates['total_error_1'], aggregates['total_error_2'], aggregates['total_error_3'],
                    ]), 'total_error_B': hmean([
                            aggregates['total_error_1'], aggregates['total_error_2'], aggregates['total_error_4'],
                    ])}
                    fix_values = sgr_df.loc[
                        sgr_df.index[0], sgr_df.columns[sgr_df.applymap(str).nunique() == 1]
                    ].to_dict()
                    subgroup_aggr_stats[f'{l0_name}_{sg_name}'] = fix_values | aggregates

                # subgroup with training and evaluation on same data (only exists for unsupervised methods)
                same_data_df = l0_df.loc[l0_df['same_variant']]
                if len(same_data_df.index) > 1:
                    process_subgroup(same_data_df, 'same')

                # subgroup with training always done on original-positive data
                other_data_df = l0_df.loc[~l0_df['same_variant']]
                process_subgroup(other_data_df, 'origin_pos')

            # # # # # # # # # #
            # GENERATE PLOTS  #
            # # # # # # # # # #
            for group in ['same_variant_grp', 'pos_origin_variant_grp']:
                df = l1_df[l1_df[group]][
                    ['label', 'prediction', 'premise_origin', 'premise_asserted', 'TRAIN_STEP_NAME.3']
                ]
                df = df.explode(['label', 'prediction'])
                g = sns.FacetGrid(df, row='TRAIN_STEP_NAME.3', col='label', margin_titles=False, ylim=(0, 1), aspect=2)

                # add a horizontal line
                for ax in g.axes.flatten():
                    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

                # Draw the plots
                args = ('premise_origin', 'prediction', 'premise_asserted')
                kwargs = {
                    'palette': 'muted',
                    'order': ['original', 'random', 'shuffled', 'no-premise'],
                    # 'hue_order': ['True', 'False']
                }
                g.map(sns.boxplot, *args, showfliers=False, **kwargs)
                g.map(sns.stripplot, *args, size=3, alpha=0.2, jitter=.2, linewidth=0.1, dodge=True, **kwargs)

                # Create custom legend elements
                legend_elements = [
                    mpatches.Patch(color=color, label=label)
                    for label, color in zip(['True', 'False'], sns.color_palette("muted"))
                ] + [mpatches.Patch(color='grey', label='N/A')]
                g.add_legend(handles=legend_elements)

                for ax in g.axes.flat:
                    # Change boxplot color
                    box_children = [child for child in ax.get_children() if isinstance(child, matplotlib.patches.PathPatch)]
                    if box_children:
                        box_children[-1].set_facecolor('grey')

                    # Change stripplot color
                    stripplot_children = [child for child in ax.get_children() if
                                          isinstance(child, matplotlib.collections.PathCollection)]
                    if stripplot_children:
                        stripplot_children[-1].set_facecolor('grey')
                        stripplot_children[-1].set_edgecolor('grey')

                    title = ax.get_title().replace(' | ', '\n| ')
                    ax.set_title(title, fontsize=9)

                g.fig.tight_layout()
                g.fig.savefig(self.work_dir.parent / f'plot_{group}_{l1_name}.pdf', format='pdf', dpi=300)

        pd.DataFrame.from_dict(subgroup_aggr_stats).T.to_csv(self.work_dir.parent / 'subgroup_aggr_stats.csv')

        return None


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
