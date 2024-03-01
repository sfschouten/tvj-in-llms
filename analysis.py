import itertools
from functools import reduce
import operator
import hashlib

import pandas as pd
import torch
import duckdb
import pandas
from duckdb import DuckDBPyConnection
from pandas import DataFrame
from tango import Step
from tango.format import TextFormat
import numpy as np
from scipy import stats

import seaborn as sns
import seaborn.objects as so
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from integrations import DuckDBFormat
from evaluate import ProbeResults

PRIMITIVES = (bool, str, int, float)


@Step.register('duckdb_builder')
class DuckDBBuilder(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = '016'

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

        def with_dependencies(step):
            yield step
            yield from step.recursive_dependencies

        info = [
            merge_dicts([
                extract_relevant_info(dependency, 'eval')
                for dependency in with_dependencies(step.kwargs['data'])
            ] + [
                extract_relevant_info(dependency, 'train')
                for dependency in with_dependencies(step.kwargs['probe'].kwargs['train_data'])
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
                'train_LoadData_prompt_name', 'train_LoadData_dataset_config_name', 'TRAIN_STEP_NAME.0',
                'eval_LoadData_prompt_name', 'eval_LoadData_dataset_config_name',
            } | probe_details]
            # level 1 groups consist of level 0 groups that also differ in probe type
            excl.append(excl[0] | {'TRAIN_STEP_NAME.4', 'train_Normalize_var_normalize', 'eval_Normalize_var_normalize'
                                    # 'eval_CreateSplits_layer_index'
                                   })
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


# @Step.register('column_rename')
# class ColumnRename(Step[DuckDBPyConnection]):
#     FORMAT = DuckDBFormat
#
#     def run(self, db: DuckDBPyConnection, table_name: str, name_map: dict):
#         for old, new in name_map.items():
#             db.sql(f'ALTER TABLE {table_name} RENAME {old} TO {new};')
#         return db


def create_figure_name(df, by_layer=True, extension='pdf'):
    model_name = df['train_AutoTokenizerLoader_pretrained_model_name_or_path'].iat[0]
    layer = df['train_CreateSplits_layer_index'].iat[0]
    group = f"{df['same_variant_grp'].iat[0]}_{df['pos_origin_variant_grp'].iat[0]}"
    file_name = f"plot_{group}_{model_name.split('/')[-1]}"
    if by_layer:
        file_name += f"_{layer}"
    file_name += f".{extension}"
    return file_name, model_name, layer


AGGR_TYPES = ['mean', 'median', 'trim_mean', '10pth', '90pth']
ERR_COLS = ['error_sv', 'error_1', 'error_2', 'error_3', 'error_4']


@Step.register('prepare_dataframe')
class PrepareDataframe(Step):
    VERSION = "002"

    def run(self, db: DuckDBPyConnection, **kwargs) -> pd.DataFrame:
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
        df['premise_asserted'] = ~df['eval_prompt'].str.endswith('-premise_negated')
        df['premise_origin'] = df['eval_config'].replace({
            'no_neutral_shuffle_premises': 'shuffled',
            'no_neutral_random_bits': 'random',
            'no_neutral': 'original'
        })

        # set the origin of -hypothesis-only prompts to 'no-premise'
        df.loc[df['eval_prompt'].str.endswith('-hypothesis_only'), 'premise_origin'] = 'no-premise'

        # create column to mark which rows were trained and evaluated on the same type of data
        df['same_variant'] = (df['eval_prompt'] == df['train_prompt']) & (df['eval_config'] == df['train_config'])

        # create groups (the positive-premise runs go in both groups)
        pos_origin_idx = (df['premise_origin'] == 'original') & (df['eval_prompt'].str.endswith('-full'))
        df['same_variant_grp'] = df['same_variant'] | pos_origin_idx
        df['pos_origin_variant_grp'] = ~df['same_variant'] | pos_origin_idx

        # add columns for error types
        aggr_err_cols = [f'{t}_{err}' for err in ERR_COLS for t in AGGR_TYPES]
        df = df.assign(**{err_col: None for err_col in ERR_COLS + aggr_err_cols})
        return df


@Step.register('calc_error_scores')
class CalculateErrorScores(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = "012"

    def run(self, df: pd.DataFrame, **kwargs) -> DuckDBPyConnection:
        subgroup_aggr_stats = {}

        aggr_err_cols = [col for col in df.columns if '_error' in col]

        # each level-1 group contains rows which have model and layer in common
        for l1_name, l1_df in df.groupby(by='data_group_l1_id'):

            # each level-0 group also have the same probe-type
            for l0_name, l0_df in l1_df.groupby(by='data_group_l0_id'):
                # select the row corresponding to the run that was evaluated on the original, positive premises
                original_pos = l0_df[(l0_df['premise_origin'] == 'original')
                                     & l0_df['eval_prompt'].str.endswith('-full')]

                def process_subgroup(sgr_df, sg_name):
                    a, o = sgr_df['premise_asserted'], sgr_df['premise_origin']
                    h_nly_i = sgr_df[sgr_df['eval_prompt'].str.endswith('-hypothesis_only')].index

                    # predictions
                    hyp_only_pred_vals = sgr_df.loc[h_nly_i].prediction.values[0]
                    o_pos_vals = original_pos.prediction.values[0]

                    # premise effect
                    pe = o_pos_vals - hyp_only_pred_vals

                    # === ERRORS ===
                    for origin, error_i in [('random', 1), ('shuffled', 2)]:
                        for asserted in [True, False]:
                            # error types 1 & 2
                            bl_i = sgr_df[(o == origin) & (a == asserted)].index
                            assert len(bl_i) == 1
                            err = abs(sgr_df.loc[bl_i].prediction.values[0] - hyp_only_pred_vals) / abs(pe)
                            df.at[bl_i[0], f'error_{error_i}'] = err

                    o_neg_i = sgr_df[(o == 'original') & ~a].index
                    assert len(o_neg_i) == 1

                    # supervised error
                    labels = original_pos.label.values[0].astype(bool)
                    sv_score_1 = labels * pe.clip(min=0)
                    sv_score_0 = ~labels * (-pe).clip(min=0)
                    df.at[o_neg_i[0], 'error_sv'] = sv_score_1 + sv_score_0

                    # error type 3
                    o_neg_vals = sgr_df.loc[o_neg_i].prediction.values[0]
                    df.at[h_nly_i[0], 'error_3'] = ((o_neg_vals - hyp_only_pred_vals) / pe).clip(min=0)

                    # error type 4
                    df.at[o_neg_i[0], 'error_4'] = abs(o_neg_vals - o_pos_vals) / abs(pe)

                    # calculate aggregates
                    for col in df.columns:
                        if col.startswith('error_'):
                            df.loc[sgr_df.index, 'median_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.median(x)
                            )
                            df.loc[sgr_df.index, 'mean_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else x.mean()
                            )
                            df.loc[sgr_df.index, 'trim_mean_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else stats.trim_mean(x, 0.05)
                            )
                            df.loc[sgr_df.index, '10pth_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.percentile(x, 10)
                            )
                            df.loc[sgr_df.index, '90pth_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.percentile(x, 90)
                            )

                    # aggregate errors
                    aggregates = df.loc[sgr_df.index, aggr_err_cols].mean(axis=0).to_dict()

                    # === PROBABILITY AVERAGES ===
                    mean_preds = {
                        'p(h|p;e)': o_pos_vals[labels].mean(),
                        'p(h|p;c)': o_pos_vals[~labels].mean(),
                        'p(h|-p;e)': o_neg_vals[labels].mean(),
                        'p(h|-p;c)': o_neg_vals[~labels].mean(),
                        'p(h)': hyp_only_pred_vals.mean(),
                    }

                    # === METRICS ===
                    # accuracy
                    metrics = {'accuracy': ((original_pos.prediction.values[0] > 0.5) == labels).mean()}
                    metrics['error_sv_0'] = np.mean(sv_score_0)
                    metrics['error_sv_1'] = np.mean(sv_score_1)

                    param_values = sgr_df.loc[
                        sgr_df.index[0], sgr_df.columns[sgr_df.applymap(str).nunique() == 1]
                    ].to_dict()  # add columns with single value for this subgroup
                    subgroup_aggr_stats[f'{l0_name}_{sg_name}'] = param_values | aggregates | mean_preds | metrics

                # subgroup with training and evaluation on same data (only exists for unsupervised methods)
                same_data_df = l0_df.loc[l0_df['same_variant']]
                if len(same_data_df.index) > 1:
                    process_subgroup(same_data_df, 'same')

                # subgroup with training always done on original-positive data
                other_data_df = l0_df.loc[~l0_df['same_variant']]
                process_subgroup(other_data_df, 'origin_pos')

        aggr_stats_df = pd.DataFrame.from_dict(subgroup_aggr_stats).T
        duckdb.sql("CREATE TABLE aggr_stats AS SELECT * FROM aggr_stats_df;")
        return duckdb.default_connection


@Step.register('strip_plot_predictions')
class StripPlotPredictions(Step):

    def run(self, df: pd.DataFrame):

        for l1_name, l1_df in df.groupby(by='data_group_l1_id'):

            for group in ['same_variant_grp', 'pos_origin_variant_grp']:
                plot_df = l1_df[l1_df[group]][
                    ['label', 'prediction', 'premise_origin', 'premise_asserted', 'TRAIN_STEP_NAME.4']
                ]
                plot_df = plot_df.explode(['label', 'prediction'])
                g = sns.FacetGrid(
                    plot_df,
                    row='TRAIN_STEP_NAME.4', col='label', margin_titles=False, ylim=(0, 1), aspect=2
                )

                # add a horizontal line
                for ax in g.axes.flatten():
                    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)

                # Draw the plots
                args = ('premise_origin', 'prediction', 'premise_asserted')
                kwargs = {
                    'palette': 'muted',
                    'order': ['original', 'random', 'shuffled', 'no-premise'],
                    'hue_order': [True, False]
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
                    if len(box_children) == 7:
                        rightmost = max((child for child in box_children),
                                        key=lambda c: c.get_path().vertices[:, 0].min())
                        rightmost.set_facecolor('grey')

                    # Change stripplot color
                    stripplot_children = [child for child in ax.get_children()
                                          if isinstance(child, matplotlib.collections.PathCollection)
                                          and len(child.get_offsets()) > 0]
                    if len(stripplot_children) == 7:
                        rightmost = max((child for child in stripplot_children),
                                        key=lambda c: c.get_offsets()[:, 0].min())
                        rightmost.set_facecolor('grey')
                        rightmost.set_edgecolor('grey')

                    title = ax.get_title().replace(' | ', '\n| ')
                    ax.set_title(title, fontsize=9)

                # create title
                filename, model_name, layer = create_figure_name(l1_df)
                g.fig.suptitle(f'{model_name}   |   Layer {layer}')
                g.fig.subplots_adjust(top=0.9)
                g.fig.savefig(self.work_dir.parent / filename, format='pdf', dpi=300)


@Step.register('calc_metric_ranks')
class CalcMetricRanks(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat

    def run(self, db: DuckDBPyConnection):
        errors = [f'error_{i}' for i in range(1, 5)] + ['error_sv']
        for t in AGGR_TYPES:
            for e in errors:
                db.sql(f'ALTER TABLE aggr_stats ADD COLUMN "{t}_{e}_rank" INT;')
            db.sql(f'ALTER TABLE aggr_stats ADD COLUMN "{t}_avg_rank" FLOAT;')

        cols = [(f'{t}_{e}', f'{t}_{e}_rank') for t, e in itertools.product(AGGR_TYPES, errors)]
        for score_name, rank_name in cols:
            db.sql(f'''
                CREATE TEMP TABLE tmp_{rank_name} AS
                    SELECT data_group_l0_id, same_variant, "{score_name}",
                        row_number() OVER (PARTITION BY data_group_l1_id, same_variant 
                                           ORDER BY "{score_name}" DESC) AS rank
                    FROM aggr_stats;
            
                UPDATE aggr_stats AS l
                SET "{rank_name}" = r.rank 
                FROM tmp_{rank_name} AS r
                WHERE r.data_group_l0_id = l.data_group_l0_id
                AND r.same_variant = l.same_variant;
                
                DROP TABLE tmp_{rank_name};
            ''')

        # calculate average ranks
        for t in AGGR_TYPES:
            rank_cols = " + ".join([f'"{t}_{e}_rank"' for e in errors])
            db.sql(f"""
                UPDATE aggr_stats
                SET "{t}_avg_rank" = ({rank_cols}) / {len(errors)};
            """)

        return db


@Step.register('plot_metrics')
class PlotMetrics(Step):
    VERSION = "008"

    def run(self, db: DuckDBPyConnection):
        df = db.sql("SELECT * FROM aggr_stats").df()

        for grp_name, grp_df in df.groupby(by=['same_variant_grp']):
            base_name, _, _ = create_figure_name(grp_df, by_layer=False)

            # line plot for each independent metric, layers on x, metric on y, method as hue
            metrics = ['accuracy']
            IND_AGGR = ['mean', 'trim_mean', 'median']
            ERR = [f'error_{i}' for i in range(1, 5)] + ['error_sv']
            metrics += [f'{t}_{e}' for e in ERR for t in IND_AGGR]
            for metric in metrics:
                plt.figure()
                ax = sns.lineplot(x="train_CreateSplits_layer_index", y=metric, hue="PROBE_type", data=grp_df)
                filename = metric + "_" + base_name
                ax.figure.savefig(self.work_dir.parent / filename, format='pdf', dpi=300)

            # add also 10pth - median - 90pth
            for e in ERR:
                plt.figure()
                plot = so.Plot(
                    grp_df,
                    x="train_CreateSplits_layer_index",
                    y=f'median_{e}', ymin=f'10pth_{e}', ymax=f'90pth_{e}',
                    color='PROBE_type'
                ).add(so.Line(linewidth=2)).add(so.Band(edgewidth=1))
                plot.save(
                    self.work_dir.parent / f'median_pth_{e}_{base_name}',
                    format='pdf', dpi=300, bbox_inches='tight'
                )


@Step.register('plot_e3_e4')
class PlotE3E4(Step):
    VERSION = "005"

    def run(self, db: DuckDBPyConnection, aggr_type: str = "median"):
        df = db.sql("SELECT * FROM aggr_stats").df()

        for grp_name, grp_df in df.groupby(by=['same_variant_grp']):
            base_name, _, _ = create_figure_name(grp_df, by_layer=False)

            # scatter plot with E3 and E4 on x and y, method as mark, and layer as color
            plt.figure()
            plot = so.Plot(
                grp_df,
                x=f'{aggr_type}_error_3', y=f'{aggr_type}_error_4', marker='PROBE_type',
                color='train_CreateSplits_layer_index'
            ).add(so.Dots())
            plot.save(
                self.work_dir.parent / ('E3_E4_scatter' + base_name),
                format='pdf', dpi=300, bbox_inches='tight'
            )

            # line plot showing log ratio of E3 to E4
            plt.figure()
            grp_df['log_ratio'] = np.log(grp_df[f'{aggr_type}_error_3'] / grp_df[f'{aggr_type}_error_4'])
            plot = so.Plot(
                grp_df,
                x='train_CreateSplits_layer_index', y='log_ratio', color='PROBE_type'
            ).add(so.Line()).layout(size=(7, 3))
            plot.save(
                self.work_dir.parent / ('log_ratio_E3_E4' + base_name),
                format='pdf', dpi=300, bbox_inches='tight'
            )


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
