#### GIN fine-tuning
split=scaffold
dataset='tox21'

CUDA_VISIBLE_DEVICES=0
for runseed in 0 1 2 3 4 5 6 7 8 9 
do
model_file=${unsup}
python finetune.py --input_model_file '' \
                   --split $split \
                   --runseed $runseed \
                   --gnn_type gin \
                   --device 6 \
                   --dataset $dataset \
                   --filename tox21_f_dc_e1_25 \
                   --lr 1e-3 \
                   --epochs 100
done