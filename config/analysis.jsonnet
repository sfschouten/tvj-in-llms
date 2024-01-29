local main = import 'main.jsonnet';

local all_steps = main.steps_model_func('llama-7b');
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
        'context_sensitivity': {
            "type": "context_sensitivity",
            db: {'ref': 'results_db'},
        }
	}
}