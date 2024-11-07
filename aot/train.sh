exp="default"
gpu_num="1"

model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_deaotl"
model="swinb_aotl"
####

## Training ##
#stage="pre"
#python tools/train.py --amp \
#	--exp_name ${exp} \
#	--stage ${stage} \
#	--model ${model} \
#	--gpu_num ${gpu_num}

stage="pre_ytb_dav"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num}