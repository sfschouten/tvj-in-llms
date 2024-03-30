DATE=$(date +'%Y-%m-%d_%H-%M-%S')


# experiment 1
tango run config/exp1.jsonnet --name "exp1_snli_$DATE" --ext-var dataset=snli
tango run config/exp1.jsonnet --name "exp1_entbank_$DATE" --ext-var dataset=entbank


# experiment 2
tango run config/exp2.jsonnet --name "exp2_llama2-7b_snli_$DATE" --ext-var dataset=snli --ext-var model=llama2-7b
tango run config/exp2.jsonnet --name "exp2_llama2-7b_entbank_$DATE" --ext-var dataset=entbank --ext-var model=llama2-7b

tango run config/exp2.jsonnet --name "exp2_llama2-13b_snli_$DATE" --ext-var dataset=snli --ext-var model=llama2-13b
tango run config/exp2.jsonnet --name "exp2_llama2-13b_entbank_$DATE" --ext-var dataset=entbank --ext-var model=llama2-13b

tango run config/exp2.jsonnet --name "exp2_olmo-7b_snli_$DATE" --ext-var dataset=snli --ext-var model=olmo-7b
tango run config/exp2.jsonnet --name "exp2_olmo-7b_entbank_$DATE" --ext-var dataset=entbank --ext-var model=olmo-7b

tango run config/exp2.jsonnet --name "exp2_olmo-7b-instruct_snli_$DATE" --ext-var dataset=snli --ext-var model=olmo-7b-instruct
tango run config/exp2.jsonnet --name "exp2_olmo-7b-instruct_entbank_$DATE" --ext-var dataset=entbank --ext-var model=olmo-7b-instruct



# comparing directions
tango run config/compare_directions.jsonnet --name "comp_dir_llama2-7b_$DATE" --ext-var model=llama2-7b
tango run config/compare_directions.jsonnet --name "comp_dir_llama2-13b_$DATE" --ext-var model=llama2-13b
tango run config/compare_directions.jsonnet --name "comp_dir_olmo-7b_$DATE" --ext-var model=olmo-7b
tango run config/compare_directions.jsonnet --name "comp_dir_olmo-7b-instruct_$DATE" --ext-var model=olmo-7b-instruct




