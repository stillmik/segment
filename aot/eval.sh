exp="default"
gpu_num="1"

model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_deaotl"
model="swinb_aotl"

stage="pre"

## Evaluation ##
dataset="davis2017"
#split="val"
#python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
#	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

dataset="davis2017"
split="test"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}