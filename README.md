# PPT

## Positional Prompt Tuning for Efficient 3D Representation Learning. [ArXiv](https://arxiv.org/abs/)

In this work, we rethink the effect of position embedding in Transformer-based point cloud representation learning methods, and present a novel method of Parameter-Efficient Fine-Tuning(PEFT) tasks based on this as well as a new form of prompt and adapter structure. With less than 5% of the trainable parameters, our method, namely PPT, outperforms its PEFT counterparts in classification tasks on ModelNet40 and ScanObjectNN datasets. Our PPT also gets better or on par results in few-shot learning on ModelNet40 and in part segmentation on ShapeNetPart.

<div  align="center">    
 <img src="./figure/pipeline.jpg" width = "666"  align=center />
</div>

## 1. Requirements
PyTorch >= 1.7.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;
### Quick Start
```
conda create -n ppt python=3.10 -y
conda activate ppt

conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## 2. Datasets

We use ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

<!-- ## 3. PPT Models
| Task              | Dataset        | Config                                                                         | Acc.       | Download                                                                                 |      
|-------------------|----------------|--------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | N.A.                                                                           | N.A.       | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth)     |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](cfgs/pointmae_configs/finetune_scan_hardest.yaml) | 85.18%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](cfgs/pointmae_configs/finetune_scan_objbg.yaml)     | 90.02%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth)   |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](cfgs/pointmae_configs/finetune_scan_objonly.yaml) | 88.29%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](cfgs/pointmae_configs/finetune_modelnet.yaml)         | 93.80%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth)  |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](cfgs/pointmae_configs/finetune_modelnet_8k.yaml)   | 94.04%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_8k.pth)  |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                                 | 86.1% mIoU | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth)     |

| Task              | Dataset    | Config                                             | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |     
|-------------------|------------|----------------------------------------------------|----------------|----------------|-----------------|-----------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](cfgs/pointmae_configs/fewshot.yaml) | 96.3 ± 2.5     | 97.8 ± 1.8     | 92.6 ± 4.1      | 95.0 ± 3.0      | 

## 4. ReCon Models
| Task              | Dataset        | Config                                                                      | Acc.       | Download                                                                                 |      
|-------------------|----------------|-----------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | N.A.                                                                        | N.A.       | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth)     |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](cfgs/recon_configs/finetune_scan_hardest.yaml) | 85.18%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth) |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](cfgs/recon_configs/finetune_scan_objbg.yaml)     | 90.02%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth)   |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](cfgs/recon_configs/finetune_scan_objonly.yaml) | 88.29%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth) |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](cfgs/recon_configs/finetune_modelnet.yaml)         | 93.80%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_1k.pth)  |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](cfgs/recon_configs/finetune_modelnet_8k.yaml)   | 94.04%     | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/modelnet_8k.pth)  |
| Part segmentation | ShapeNetPart   | [segmentation](./segmentation)                                              | 86.1% mIoU | [here](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth)     |

| Task              | Dataset    | Config                                          | 5w10s Acc. (%) | 5w20s Acc. (%) | 10w10s Acc. (%) | 10w20s Acc. (%) |     
|-------------------|------------|-------------------------------------------------|----------------|----------------|-----------------|-----------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](cfgs/recon_configs/fewshot.yaml) | 96.3 ± 2.5     | 97.8 ± 1.8     | 92.6 ± 4.1      | 95.0 ± 3.0      |  -->

## 3. PPT Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

<!-- ## 6. Visualization

Visulization of pre-trained model on ShapeNet validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/pretrain.yaml --exp_name <name>
``` -->

## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [ICCV23-IDPT](https://github.com/zyh16143998882/ICCV23-IDPT?tab=readme-ov-file)

## Reference

```

```
