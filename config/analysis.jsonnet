local dataset = std.extVar('dataset');

local utils = import 'utils.libsonnet';
local common = import 'common.libsonnet';
local data = import 'data.libsonnet';
local models = import 'models.libsonnet';


local all_steps_l = [
    common.steps_model_func(model_key, models[model_key], data[dataset])
    for model_key in std.objectFields(models)
];

local train_steps = utils.join_objects([s['train_steps'] for s in all_steps_l]);
local eval_steps = utils.join_objects([s['eval_steps'] for s in all_steps_l]);
local model_steps = utils.join_objects([s['model_step'] for s in all_steps_l]);
local data_steps = utils.join_objects([s['data_steps'] for s in all_steps_l]);
{
	"steps": model_steps + data_steps + train_steps + eval_steps + {
	    'results_db': {
	        "type": "duckdb_builder",
	        result_inputs: std.objectKeysValues(eval_steps),
	        results: std.map(
	            function(key) {"ref": key},
	            std.objectFields(eval_steps)
	        ),
	    },
        'df': {
            type: 'prepare_dataframe',
            db: {ref: 'results_db'},
        },
//        'strip_plots': {
//            type: 'strip_plot_predictions',
//            df: {ref: 'df'},
//        },
        'metrics': {
            type: "calc_error_scores",
            df: {ref: 'df'},
        },
        'rank_metrics': {
            type: 'calc_metric_ranks',
            db: {ref: 'metrics'},
        },
        'metric_plots': {
            type: 'plot_metrics',
            db: {ref: 'metrics'}
        },
        'e3_e4_plots': {
            type: 'plot_e3_e4',
            db: {ref: 'metrics'}
        }
	}
}