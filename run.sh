export CUDA_VISIBLE_DEVICES=0,1,2
name=base
python train_ms.py -c configs/$name.json -m $name
