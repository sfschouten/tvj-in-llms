import re

import numpy as np
import pandas as pd
import torch
import duckdb

from tango import Step

from duckdb import DuckDBPyConnection

import seaborn.objects as so

from integrations import DuckDBFormat
from evaluate import ProbeResults

PRIMITIVES = (bool, str, int, float)

METHOD_MAP = {
    'lr_sklearn': 'LR',
    # 'lm_head_baseline': 'LM-head',
    'mass_mean': 'MMP',
    # 'ccs_gd': 'CCS',
    'ccr': 'CCR',
}


@Step.register('duckdb_builder2')
class DuckDBBuilder2(Step[DuckDBPyConnection]):
    FORMAT = DuckDBFormat
    VERSION = '001'

    def run(self, result_inputs: list, results: list[ProbeResults]) -> DuckDBPyConnection:
        dependencies_by_name = {step.name: step for step in self.dependencies}
        input_steps = [dependencies_by_name[_input.params['key']] for _input in result_inputs]

        def normalize(arraylike):
            if arraylike is None:
                return None
            return torch.tensor(arraylike).squeeze().float().tolist()

        results_by_step_name = {step.name: result for step, result in zip(input_steps, results)}
        results_and_properties = []
        for name, results in results_by_step_name.items():
            if 'INTERVENED' in name:
                intervention, layer, _ = name.split('|')
                m = re.match(r'INTERVENED_on\[.*\]_with\[(.*)\]', intervention)
                intervention_probe = m.group(1)
                data, model, _, _, probe_method = intervention_probe.split(',')
                setting = 'intervened'
            else:
                data, model, layer, _, probe_method = name.split('|')
                setting = 'original'
            entry = {
                'data': data, 'model': model, 'layer': layer, 'probe_method': probe_method, 'setting': setting,
                'labels': normalize(results[1]), 'predictions': normalize(results[2])
            }
            results_and_properties.append(entry)

        df = pd.DataFrame.from_records(results_and_properties)

        # store in duckdb
        db = duckdb.connect()
        db.execute("SET GLOBAL pandas_analyze_sample=100000")
        db.sql('CREATE TABLE results AS SELECT * FROM df')

        return db


@Step.register('causal_analysis')
class CausalAnalysis(Step):
    VERSION = "005"

    def run(self, results_db: DuckDBPyConnection):

        df = results_db.sql('''
            SELECT model, layer, probe_method, 
                any_value(labels) AS labels, 
                list(setting) AS settings,
                first(predictions) AS intervened_predictions, 
                last(predictions) AS original_predictions
            FROM (SELECT * FROM results ORDER BY setting)
            GROUP BY model, layer, probe_method
        ''').df()
        df['probe_method'] = df['probe_method'].replace(METHOD_MAP)

        df['labels'] = df.apply(lambda x: np.array(x.labels).astype(bool), axis=1)
        df['layer'] = df.apply(lambda x: int(x.layer.replace('layer', '')), axis=1)

        df['original_predictions'] = df.apply(lambda x: np.array(x.original_predictions), axis=1)
        df['intervened_predictions'] = df.apply(lambda x: np.array(x.intervened_predictions), axis=1)

        df['mean_original_0'] = df.apply(lambda x: x.original_predictions[~x.labels].mean(), axis=1)
        df['mean_intervened_0'] = df.apply(lambda x: x.intervened_predictions[~x.labels].mean(), axis=1)
        df['mean_original_1'] = df.apply(lambda x: x.original_predictions[x.labels].mean(), axis=1)
        df['mean_intervened_1'] = df.apply(lambda x: x.intervened_predictions[x.labels].mean(), axis=1)

        df['differences'] = df['intervened_predictions'] - df['original_predictions']
        df['differences_0'] = df.apply(lambda x: x.differences[~x.labels], axis=1)
        df['differences_1'] = df.apply(lambda x: x.differences[x.labels], axis=1)
        df['mean_difference_0'] = df.apply(lambda x: x.differences_0.mean(), axis=1)
        df['mean_difference_1'] = df.apply(lambda x: x.differences_1.mean(), axis=1)
        df['median_difference_0'] = df.apply(lambda x: np.median(x.differences_0), axis=1)
        df['median_difference_1'] = df.apply(lambda x: np.median(x.differences_1), axis=1)

        for aggr in ['mean', 'median']:
            p = so.Plot(data=df, x='layer', color='probe_method') \
                .add(so.Line(linestyle='dashed'), y=f'{aggr}_difference_0') \
                .add(so.Line(linestyle='solid'), y=f'{aggr}_difference_1') \
                .layout(size=(7, 4)) \
                .limit(y=(-10, 10)) \
                .scale(color=so.Nominal(order=sorted(METHOD_MAP.values()))) \
                .label(x='Layer', y='Mean difference', color='Method')
            p.save(
                self.work_dir.parent / f'causal_plot_{aggr}_difference.pdf',
                format='pdf', dpi=300, bbox_inches='tight'
            )

        for l in [0, 1]:
            aggr = 'mean'
            p = so.Plot(data=df, x='layer', color='probe_method') \
                .add(so.Line(linestyle='solid'), y=f'{aggr}_original_{l}') \
                .add(so.Line(linestyle='dashed'), y=f'{aggr}_intervened_{l}') \
                .layout(size=(7, 4)) \
                .limit(y=(0, 1)) \
                .scale(color=so.Nominal(order=sorted(METHOD_MAP.values()))) \
                .label(x='Layer', y='Mean probability', color='Method')
            p.save(
                self.work_dir.parent / f'causal_plot_mean_{l}.pdf',
                format='pdf', dpi=300, bbox_inches='tight'
            )
