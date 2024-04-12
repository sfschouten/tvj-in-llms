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
import numpy as np
from scipy import stats

import seaborn as sns
import seaborn.objects as so
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from beliefprobing.integrations import DuckDBFormat
from beliefprobing.probes.beliefprobe import ProbeResults

PRIMITIVES = (bool, str, int, float)


@Step.register('duckdb_builder')
class DuckDBBuilder(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = '018'

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
                | {f'EVAL_STEP_NAME_@.0': step.name.split('@')[-1].split('|')[0]}
                | {
                    f'TRAIN_STEP_NAME.{i}': part
                    for i, part in enumerate(result[0].train_step_name.split('|'))
                }
            ])
            for result, step in zip(results, input_steps)
        ]

        # create groups of results to analyse together.
        for info_dict in info:
            keys = set(info_dict.keys()) - {'EVAL_STEP_NAME', 'EVAL_STEP_NAME_@.0'}
            probe_details = {key for key in info_dict.keys() if key.startswith('PROBE')}
            # results that only differ in the data variant used for training/evaluation belong to the same level 0 group
            excl = [{
                'train_LoadData_prompt_name', 'train_LoadData_dataset_config_name', 'TRAIN_STEP_NAME.0',
                'eval_LoadData_prompt_name', 'eval_LoadData_dataset_config_name',
            } | probe_details]
            # level 1 groups consist of level 0 groups that also differ in probe type
            excl.append(excl[0] | {
                'TRAIN_STEP_NAME.4', 'train_Normalize_var_normalize', 'eval_Normalize_var_normalize'
            })
            # level 2 groups also ignore the layer, so all level 1 trained/evaluated on the same model are merged
            excl.append(excl[1] | {
                'TRAIN_STEP_NAME.2', 'eval_CreateSplits_layer_index', 'train_CreateSplits_layer_index'
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
        db.execute("SET GLOBAL pandas_analyze_sample=100000")
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
    group = f"train@{df['subgroup'].iat[0]}"
    file_name = f"plot_{group}_{model_name.split('/')[-1]}"
    if by_layer:
        file_name += f"_{layer}"
    file_name += f".{extension}"
    return file_name, model_name, layer


AGGR_TYPES = ['mean', 'median', 'trim_mean', '20pth', '80pth']
# ERR_COLS_1 are sensitive to outliers, require aggregation types other than the mean
ERR_COLS_1 = ['error_1', 'error_2', 'error_3', 'error_4', 'error_3+4']
ERR_COLS_2 = ['error_sv']
ERR_COLS = ERR_COLS_1 + ERR_COLS_2
OTHER_METRICS = ['accuracy@no_prem', 'accuracy@pos_prem', 'premise_sensitivity', 'rel_error_sv']
METHOD_MAP = {
    'lr_sklearn': 'LR',
    'lm_head_baseline': 'LM-head',
    'mass_mean': 'MMP',
    'ccs_gd': 'CCS',
    'ccr': 'CCR',
}
Y_LIMITS = {
    'error_1': (0, 2), 'error_2': (0, 2), 'error_3': (0, 2), 'error_4': (0, 2), 'error_3+4': (1, 3),
    'error_sv': (0, 0.6), 'accuracy@pos_prem': (0, 1), 'accuracy@no_prem': (0, 1), 'premise_sensitivity': (0, 0.6),
    'rel_error_sv': (0, 1),
}
LEGEND_LOC = {
    'error_1': (.86, .6), 'error_2': (.86, .6), 'error_3': (.86, .6), 'error_4': (.86, .6),
    'error_3+4': (.86, .6), 'error_sv': (.86, .6), 'accuracy@pos_prem': (.86, .07), 'accuracy@no_prem': (.86, .07),
    'premise_sensitivity': (-0.02, .6), 'rel_error_sv': (.86, .07)
}
GROUPS = ['no_prem_grp', 'pos_prem_grp', 'combined_grp', 'same_variant_grp']


@Step.register('prepare_dataframe')
class PrepareDataframe(Step):
    VERSION = "009"

    def run(self, db: DuckDBPyConnection, **kwargs) -> pd.DataFrame:
        df: DataFrame = db.sql("SELECT * FROM results").df()

        assert df['label'].apply(str).nunique() == 1, 'Labels/samples not the same, results incomparable.'

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
        df['premise_asserted'] = ~df['EVAL_STEP_NAME_@.0'].str.endswith('neg_prem')
        df['premise_origin'] = df['EVAL_STEP_NAME_@.0'].str.extract(r'\w-([a-z]*)_.*')
        df['PROBE_type'] = df['PROBE_type'].replace(METHOD_MAP)

        # create groups (the positive-premise runs go in both groups)
        for grp in GROUPS:
            df[grp] = False
        df['combined_grp'] |= df['TRAIN_STEP_NAME.0'].str.endswith('-combined')
        df['pos_prem_grp'] |= df['TRAIN_STEP_NAME.0'].str.endswith('-original_pos_prem')
        df['no_prem_grp'] |= df['TRAIN_STEP_NAME.0'].str.endswith('-no_prem')
        df['same_variant_grp'] |= df['TRAIN_STEP_NAME.0'] == df['EVAL_STEP_NAME_@.0']

        # add columns for error types
        aggr_err_cols = [f'{t}_{err}' for err in ERR_COLS for t in AGGR_TYPES]
        df = df.assign(**{err_col: None for err_col in ERR_COLS + aggr_err_cols})
        return df


@Step.register('calc_error_scores')
class CalculateErrorScores(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = "019"

    def run(self, df: pd.DataFrame, **kwargs) -> DuckDBPyConnection:
        subgroup_aggr_stats = {}

        aggr_err_cols = [col for col in df.columns if '_error' in col]

        # each level-1 group contains rows which have model and layer in common
        for l1_name, l1_df in df.groupby(by='data_group_l1_id'):

            # each level-0 group also have the same probe-type
            for l0_name, l0_df in l1_df.groupby(by='data_group_l0_id'):

                def process_subgroup(sgr_df, sg_name):
                    a, o = sgr_df['premise_asserted'], sgr_df['premise_origin']
                    h_nly_i = sgr_df[sgr_df['premise_origin'] == 'no'].index
                    original_pos = sgr_df[sgr_df['EVAL_STEP_NAME_@.0'].str.endswith('-original_pos_prem')]

                    # predictions
                    hyp_only_pred_vals = sgr_df.loc[h_nly_i].prediction.values[0]
                    o_pos_vals = original_pos.prediction.values[0]

                    # premise effect
                    pe = o_pos_vals - hyp_only_pred_vals

                    # === ERRORS ===
                    for origin, error_i in [('random', 1), ('shuffle', 2)]:
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
                    sv_score_1 = labels * (-pe).clip(min=0)
                    sv_score_0 = ~labels * pe.clip(min=0)
                    df.at[o_neg_i[0], 'error_sv'] = sv_score_1 + sv_score_0

                    # error type 3
                    o_neg_vals = sgr_df.loc[o_neg_i].prediction.values[0]
                    df.at[h_nly_i[0], 'error_3'] = ((o_neg_vals - hyp_only_pred_vals) / pe).clip(min=0)

                    # error type 4
                    df.at[o_neg_i[0], 'error_4'] = abs(o_neg_vals - o_pos_vals) / abs(pe)

                    # error type 3 + 4
                    df.at[o_neg_i[0], 'error_3+4'] = df.at[h_nly_i[0], 'error_3'] + df.at[o_neg_i[0], 'error_4']

                    # calculate aggregates
                    for col in df.columns:
                        if col in ERR_COLS:
                            df.loc[sgr_df.index, 'mean_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else x.mean()
                            )
                        if col in ERR_COLS_1:
                            df.loc[sgr_df.index, 'median_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.median(x)
                            )
                            df.loc[sgr_df.index, 'trim_mean_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else stats.trim_mean(x, 0.2)
                            )
                            df.loc[sgr_df.index, '20pth_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.percentile(x, 20)
                            )
                            df.loc[sgr_df.index, '80pth_' + col] = df.loc[sgr_df.index, col].apply(
                                lambda x: pd.NA if x is None else np.percentile(x, 80)
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
                    metrics = {
                        'accuracy@pos_prem': ((original_pos.prediction.values[0] > 0.5) == labels).mean(),
                        'accuracy@no_prem': ((hyp_only_pred_vals > 0.5) == labels).mean(),
                        'premise_sensitivity': np.mean(abs(pe)),
                        'mean_premise_effect': np.mean(pe),
                        'error_sv_0': np.mean(sv_score_0),
                        'error_sv_1': np.mean(sv_score_1)
                    }
                    metrics['rel_error_sv'] = np.mean(sv_score_1 + sv_score_0) / metrics['premise_sensitivity']

                    param_values = sgr_df.loc[
                        sgr_df.index[0], sgr_df.columns[sgr_df.map(str).nunique() == 1]
                    ].to_dict()  # add columns with single value for this subgroup
                    param_values |= {'subgroup': sg_name}
                    subgroup_aggr_stats[f'{l0_name}_{sg_name}'] = param_values | aggregates | mean_preds | metrics

                # subgroup with training and evaluation on same data (only exists for unsupervised methods)
                same_data_df = l0_df.loc[l0_df['same_variant_grp']]
                if len(same_data_df.index) > 2:
                    process_subgroup(same_data_df, 'same')

                # subgroup with training on combination of variants (only exists for unsupervised methods)
                combined_data_df = l0_df.loc[l0_df['combined_grp']]
                if len(combined_data_df.index) > 0:
                    process_subgroup(combined_data_df, 'combined')

                # subgroup with training always done on data without premises
                no_prem_data_df = l0_df.loc[l0_df['no_prem_grp']]
                if len(no_prem_data_df.index) > 0:
                    process_subgroup(no_prem_data_df, 'no-prem')

                # subgroup with training always done on original-positive data
                pos_prem_data_df = l0_df.loc[l0_df['pos_prem_grp']]
                if len(pos_prem_data_df.index) > 0:
                    process_subgroup(pos_prem_data_df, 'pos-prem')

        aggr_stats_df = pd.DataFrame.from_dict(subgroup_aggr_stats).T
        duckdb.sql("CREATE TABLE aggr_stats AS SELECT * FROM aggr_stats_df;")
        return duckdb.default_connection


@Step.register('strip_plot_predictions')
class StripPlotPredictions(Step):

    def run(self, df: pd.DataFrame):

        for l1_name, l1_df in df.groupby(by='data_group_l1_id'):

            for group in GROUPS:
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
    VERSION = "010"

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
                    SELECT data_group_l0_id, subgroup, "{score_name}",
                        row_number() OVER (PARTITION BY data_group_l2_id
                                           ORDER BY "{score_name}" ASC) AS rank
                    FROM aggr_stats
                    WHERE PROBE_type != 'LM-head' 
                    OR eval_CreateSplits_layer_index = 1;
            
                UPDATE aggr_stats AS l
                SET "{rank_name}" = r.rank 
                FROM tmp_{rank_name} AS r
                WHERE r.data_group_l0_id = l.data_group_l0_id
                AND r.subgroup = l.subgroup;
                
                DROP TABLE tmp_{rank_name};
            ''')

        # calculate average ranks
        for t in AGGR_TYPES:
            rank_cols = " + ".join([f'"{t}_{e}_rank"' for e in errors])
            db.sql(f"""
                UPDATE aggr_stats
                SET "{t}_avg_rank" = ({rank_cols}) / {len(errors)};
            """)

        def query(by, m):
            return f"""
                SELECT data_group_l2_id, PROBE_type, subgroup, "TRAIN_STEP_NAME.1" AS model, 
                ARG_{m}(eval_CreateSplits_layer_index, {by}) AS layer, 
                ARG_{m}("accuracy@pos_prem", {by}) AS accuracy_pos_prem, 
                ARG_{m}(trim_mean_avg_rank, {by}) AS trim_mean_avg_rank,
                ARG_{m}("p(h|p;e)", {by}), ARG_{m}("p(h|-p;e)", {by}), 
                ARG_{m}("p(h)", {by}), 
                ARG_{m}("p(h|-p;c)", {by}), ARG_{m}("p(h|p;c)", {by}), 
                ARG_{m}(trim_mean_error_1, {by}), ARG_{m}(trim_mean_error_2, {by}),
                ARG_{m}(trim_mean_error_3, {by}), ARG_{m}(trim_mean_error_4, {by})
                FROM aggr_stats
                GROUP BY data_group_l2_id,  "TRAIN_STEP_NAME.1", PROBE_type, subgroup
            """

        # create final tables
        table_df = db.sql(
            query('"accuracy@pos_prem"', 'MAX')
            + "\nUNION ALL\n" +
            query('trim_mean_avg_rank', 'MIN')
        ).df()

        for name, final_df in table_df.groupby(by=['data_group_l2_id']):
            final_df = final_df.sort_values(by=['subgroup', 'PROBE_type'])
            final_df.to_csv(self.work_dir.parent / f'final_table_{name}.csv')

        return db


@Step.register('plot_metrics')
class PlotMetrics(Step):
    VERSION = "032"

    def run(self, db: DuckDBPyConnection):
        df = db.sql("SELECT * FROM aggr_stats").df()

        for grp_name, grp_df in df.groupby(by=['data_group_l2_id', 'subgroup']):
            base_name, _, _ = create_figure_name(grp_df, by_layer=False)

            # line plot for each independent metric, layers on x, metric on y, method as hue
            IND_AGGR = ['mean', 'trim_mean', 'median']
            metrics = [(m, m) for m in OTHER_METRICS]
            metrics += [(f'{t}_{e}', e) for e in ERR_COLS for t in IND_AGGR]
            for metric, e in metrics:
                plt.figure()
                filename = metric + "_" + base_name
                plot = so.Plot(
                    grp_df,
                    x='train_CreateSplits_layer_index', y=metric, color='PROBE_type'
                ).layout(size=(7, 4)) \
                 .add(so.Line(marker='.')) \
                 .scale(color=so.Nominal(order=sorted(METHOD_MAP.values()))) \
                 .label(x='Layer', y=metric.replace('_', ' ').capitalize(), color='Method') \
                 .limit(y=Y_LIMITS[e]) \
                 .theme({'font.size': 18, 'legend.loc': 'best'}) \
                 .plot(pyplot=True)
                plot._figure.legends[0].set_bbox_to_anchor(plot._figure.axes[0].get_position())
                plot._figure.legends[0].set_loc(LEGEND_LOC[e])
                plot.save(self.work_dir.parent / filename, format='pdf', dpi=300, bbox_inches='tight')

            # add also 20pth - median - 80pth
            for e in ERR_COLS:
                plt.figure()
                plot = so.Plot(
                    grp_df,
                    x="train_CreateSplits_layer_index",
                    y=f'median_{e}', ymin=f'20pth_{e}', ymax=f'80pth_{e}', color='PROBE_type',
                ).layout(size=(7, 4)) \
                 .add(so.Line(linewidth=2, marker='.')) \
                 .add(so.Band(edgewidth=1)) \
                 .scale(color=so.Nominal(order=sorted(METHOD_MAP.values()))) \
                 .limit(y=Y_LIMITS[e]) \
                 .label(x='Layer', color='Method') \
                 .theme({'font.size': 18}) \
                 .plot(pyplot=True)
                plot._figure.legends[0].set_bbox_to_anchor(plot._figure.axes[0].get_position())
                plot._figure.legends[0].set_loc(LEGEND_LOC[e])
                plot.save(
                    self.work_dir.parent / f'median_pth_{e}_{base_name}', format='pdf', dpi=300, bbox_inches='tight'
                )


@Step.register('plot_e3_e4')
class PlotE3E4(Step):
    VERSION = "035"

    def run(self, db: DuckDBPyConnection, aggr_type: str = "trim_mean"):
        df = db.sql("SELECT * FROM aggr_stats").df()

        for grp_name, grp_df in df.groupby(by=['data_group_l2_id', 'subgroup']):
            base_name, _, _ = create_figure_name(grp_df, by_layer=False)

            # scatter plot with E3 and E4 on x and y, method as mark, and layer as color
            plt.figure()
            plot = so.Plot(
                grp_df,
                x=f'{aggr_type}_error_3', y=f'{aggr_type}_error_4', marker='PROBE_type',
                color='train_CreateSplits_layer_index',
            ).add(so.Dots()) \
             .layout(size=(7, 4)) \
             .label(x=aggr_type + ' E3', y=aggr_type + ' E4', color='Layer', marker='Method') \
             .scale(marker=so.Nominal(order=sorted(METHOD_MAP.values()))) \
             .limit(x=Y_LIMITS['error_3'], y=Y_LIMITS['error_4']) \
             .theme({'font.size': 18}) \
             .plot(pyplot=True)
            plot._figure.legends[0].set_bbox_to_anchor(plot._figure.axes[0].get_position())
            plot._figure.legends[0].set_loc((0.82, 0.07))
            plot.save(
                self.work_dir.parent / ('E3_E4_scatter' + base_name), format='pdf', dpi=300, bbox_inches='tight'
            )

            # line plot showing log ratio of E3 to E4
            plt.figure()
            grp_df['log_ratio'] = np.log(grp_df[f'{aggr_type}_error_3'] / grp_df[f'{aggr_type}_error_4'])
            plot = so.Plot(
                grp_df,
                x='train_CreateSplits_layer_index', y='log_ratio', color='PROBE_type',
            ).add(so.Line(marker='.')) \
             .layout(size=(7, 4)) \
             .label(x='Layer', y=f'Log(E3/E4)', color='Method') \
             .scale(color=so.Nominal(order=sorted(METHOD_MAP.values()))) \
             .limit(y=(-5, 3)) \
             .theme({'font.size': 18}) \
             .plot(pyplot=True)
            plot._figure.legends[0].set_bbox_to_anchor(plot._figure.axes[0].get_position())
            plot._figure.legends[0].set_loc((-0.02, 0.07))
            plot.save(
                self.work_dir.parent / ('log_ratio_E3_E4' + base_name), format='pdf', dpi=300, bbox_inches='tight'
            )
