log_path=logs/base
checkpoint=G_91000.pth
rm $log_path/result/*
python inference.py --txt sub1.txt --checkpoint_path $log_path/$checkpoint --config configs/base.json --save_path $log_path/result
