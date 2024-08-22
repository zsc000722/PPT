python main.py --ckpts ../pointmae.pth --epoch 300 --log_dir new_bs64_lr2e-4_mae_allopen_128mlp_0 --gpu $1\
 --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal  --seed 0 --batch_size 32\
 --learning_rate 1e-4
