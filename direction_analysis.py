import pandas as pd
import torch
import duckdb

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

from tango import Step

from beliefprobing.probes.beliefprobe import BeliefProbe


@Step.register('direction_similarity')
class DirectionSimilarity(Step[str]):
    VERSION = "008"

    def run(self, result_inputs: list, probes: list[BeliefProbe]) -> str:
        # calculate cosine similarity between pairs of directions

        dependencies_by_name = {step.name: step for step in self.dependencies}
        input_steps = [dependencies_by_name[_input.params['key']] for _input in result_inputs]

        def normalize(arraylike):
            if arraylike is None:
                return None
            return torch.tensor(arraylike).squeeze().float().tolist()

        probes_by_step_name = {step.name: result for step, result in zip(input_steps, probes)}
        directions_and_properties = []
        for name, probe in probes_by_step_name.items():
            data, model, layer, _, probe_method = name.split('|')
            layer = int(layer.replace('layer', ''))
            dataset, variant = data.split('-')
            direction = probe.direction
            if direction is None:
                continue
            direction *= probe.sign
            entry = {
                'dataset': dataset, 'data_variant': variant, 'model': model, 'layer': layer,
                'probe_method': probe_method, 'direction': normalize(direction)
            }
            directions_and_properties.append(entry)

        df = pd.DataFrame.from_records(directions_and_properties)
        duckdb.sql('CREATE TABLE directions AS SELECT * FROM df')
        duckdb.sql("""
            CREATE TABLE comparisons AS SELECT 
                d1.dataset AS dataset1, d2.dataset AS dataset2,
                d1.data_variant AS data_variant1, d2.data_variant AS data_variant2, 
                d1.model AS model1, d2.model AS model2,
                d1.layer AS layer1, d2.layer AS layer2, 
                d1.probe_method AS probe_method1, d2.probe_method AS probe_method2, 
                list_cosine_similarity(d1.direction, d2.direction) AS similarity
            FROM directions AS d1
            CROSS JOIN directions AS d2
        """)

        # variables: layers (~28-36), methods (~3-5), data (2)

        # 1. similarity across datasets
        df1 = duckdb.sql("""
            SELECT * FROM comparisons
            WHERE data_variant1 = data_variant2
            AND layer1 = layer2
            AND probe_method1 = probe_method2
            AND dataset1 != dataset2
        """).df()
        plot1 = so.Plot(df1, x='layer1', y='similarity', color='probe_method1') \
                  .add(so.Line()) \
                  .facet(row='data_variant1') \
                  .limit(y=(-1, 1)) \
                  .layout(size=(7, 25))
        plot1.save(self.work_dir.parent / 'plot1.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # 2. similarity across layers
        #  A.  x: layers,  y: layers,  facets: methods, data
        df2 = duckdb.sql("""
            SELECT * FROM comparisons
            WHERE data_variant1 = data_variant2
            AND probe_method1 = probe_method2
            AND dataset1 = dataset2
        """).df()
        CMAP = 'viridis'
        for grp_name, grp_df in df2.groupby(by=['dataset1', 'data_variant1', 'probe_method1']):
            pivot_table = grp_df.pivot(columns='layer1', index='layer2', values='similarity')
            plt.figure()
            sns.heatmap(pivot_table, cmap=CMAP)
            plt.savefig(self.work_dir.parent / f'plot2_heatmap_[{"|".join(grp_name)}].pdf', format='pdf', dpi=300)

        #  B.  x: layers,  y: layers,  aggr: methods, data
        pivot_table = df2.pivot_table(columns='layer1', index='layer2', values='similarity', aggfunc='mean')
        plt.figure()
        sns.heatmap(pivot_table, cmap=CMAP)
        plt.savefig(self.work_dir.parent / f'plot2_heatmap_mean.pdf', format='pdf', dpi=300)

        # 3. similarity across methods
        # A.  x: layers,  y: method-pairs+avg,  facets: data
        df3 = duckdb.sql("""
            SELECT * FROM comparisons
            WHERE data_variant1 = data_variant2
            AND dataset1 = dataset2
            AND layer1 = layer2
            AND probe_method1 > probe_method2 
        """).df()
        df3['method_pair'] = df3['probe_method1'] + "-" + df3['probe_method2']
        plot3 = so.Plot(df3, x='layer1', y='similarity', color='method_pair') \
                  .add(so.Line()) \
                  .facet(row='data_variant1', col='dataset1') \
                  .limit(y=(-1, 1)) \
                  .layout(size=(14, 5 * df3['data_variant1'].nunique()))
        plot3.save(self.work_dir.parent / 'plot3.pdf', format='pdf', dpi=300, bbox_inches='tight')

        # 4. similarity across variants
        #  x: layers, y: similarity, facets: dataset (cols) variant-pair (row)
        df4 = duckdb.sql("""
            SELECT * FROM comparisons
            WHERE dataset1 = dataset2
            AND layer1 = layer2
            AND probe_method1 = probe_method2
            AND data_variant1 > data_variant2
        """).df()
        df4['variant-pair'] = df4['data_variant1'] + '-' + df4['data_variant2']
        plot4 = so.Plot(df4, x='layer1', y='similarity', color='probe_method1') \
                  .add(so.Line()) \
                  .facet(row='variant-pair', col='dataset1') \
                  .limit(y=(-1, 1)) \
                  .layout(size=(14, 5 * df4['variant-pair'].nunique()))
        plot4.save(self.work_dir.parent / 'plot4.pdf', format='pdf', dpi=300, bbox_inches='tight')
