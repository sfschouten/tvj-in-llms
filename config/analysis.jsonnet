local main = import 'main.jsonnet';

local all_steps = main.steps_model_func('llama-7b', std.range(1, 32));
//local all_steps = main.steps_model_func('llama-7b', [1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 27, 29, 31]);
//local all_steps = main.steps_model_func('llama-7b', [16, 17]);
local other_steps = all_steps['other_steps'];
local train_steps = all_steps['train_steps'];
local eval_steps = all_steps['eval_steps'];

{
	"steps": other_steps + train_steps + eval_steps + {
	    'results_db': {
	        "type": "duckdb_builder",
	        result_inputs: std.objectKeysValues(eval_steps),
	        results: std.map(
	            function(key) {"ref": key},
	            std.objectFields(eval_steps)
	        ),
	    },
//	    'direction_similarites': {
//	        "type": "direction_similarity",
//	        db: {'ref': 'results_db'},
//	    },
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