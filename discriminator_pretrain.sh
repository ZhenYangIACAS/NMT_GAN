export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

python discriminator_pretrain.py  -c ./configs/config_discriminator_pretrain.yaml
