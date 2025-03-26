# Mini-Splatting2：通过激进的高斯密度化在几分钟内构建360度场景

这是 **Mini-Splatting2** 的官方实现，一种在高斯 splatting 上下文中的点云重建工作。通过激进的高斯密度化，我们的算法能够在几分钟内实现快速场景优化。
有关技术细节，请参阅：

**Mini-Splatting2：通过激进的高斯密度化在几分钟内构建360度场景**  <br />
方光驰，王兵.<br />
**[[论文](https://arxiv.org/pdf/2411.12788)]** <br />

<p align="center"> <img src="./assets/teaser.jpg" width="100%"> </p>

### (1) 设置
此代码已在 Python 3.8、torch 1.12.1、CUDA 11.6 上测试通过。

- 克隆仓库
```
git clone git@github.com:fatPeter/mini-splatting2.git && cd mini-splatting2
```
- 设置 Python 环境
```
conda create -n mini_splatting2 python=3.8
conda activate mini_splatting2
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

- 下载数据集：[Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [T&T+DB COLMAP](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)。

### (2) Mini-Splatting2（稀疏高斯）

Mini-Splatting2 的训练脚本位于 `msv2` 中：
```
cd msv2
```
- 使用训练/测试分割进行训练：
```
# mipnerf360 室外
python train.py -s <数据集路径> -m <模型路径> -i images_4 --eval --imp_metric outdoor --config_path ../config/fast
# mipnerf360 室内
python train.py -s <数据集路径> -m <模型路径> -i images_2 --eval --imp_metric indoor --config_path ../config/fast
# t&t
python train.py -s <数据集路径> -m <模型路径> --eval --imp_metric outdoor --config_path ../config/fast
# db
python train.py -s <数据集路径> -m <模型路径> --eval --imp_metric indoor --config_path ../config/fast
```

- 修改后的 full_eval 脚本：
```
python full_eval.py -m360 <mipnerf360 文件夹> -tat <tanks and temples 文件夹> -db <deep blending 文件夹>
```

### (3) Mini-Splatting2-D（密集高斯）

Mini-Splatting2-D 的训练脚本位于 `msv2_d` 中：
```
cd msv2_d
```
- 使用训练/测试分割进行训练：
```
# mipnerf360 室外
python train.py -s <数据集路径> -m <模型路径> -i images_4 --eval --imp_metric outdoor --config_path ../config/fast
# mipnerf360 室内
python train.py -s <数据集路径> -m <模型路径> -i images_2 --eval --imp_metric indoor --config_path ../config/fast
# t&t
python train.py -s <数据集路径> -m <模型路径> --eval --imp_metric outdoor --config_path ../config/fast
# db
python train.py -s <数据集路径> -m <模型路径> --eval --imp_metric indoor --config_path ../config/fast
```

- 修改后的 full_eval 脚本：
```
python full_eval.py -m360 <mipnerf360 文件夹> -tat <tanks and temples 文件夹> -db <deep blending 文件夹>
```

### (4) 密集点云重建

此实现直接支持密集点云重建：

```
# 类似于 train.py (-i images_4/images_2, --imp_metric outdoor/indoor)
# 输出 ply 文件保存在 ./teaser 中
python teaser.py -s <数据集路径> -m <模型路径> -i images_4 --eval --imp_metric outdoor
```

**致谢。** 本项目基于 [Mini-Splatting](https://github.com/fatPeter/mini-splatting)、[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 和 [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs) 构建。

```
CUDA_VISIBLE_DEVICES=1 python msv2/train.py -s /workspace/2dgs/output -m /workspace/2dgs/output/3dgs --imp_metric outdoor --config_path /workspace/wb-test/mini-splatting2/config/fast
```

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python teaser.py -s /workspace/2dgs/output -m /workspace/2dgs/output/3dgs  --imp_metric outdoor --depth_reinit_iter 3000 --simp_iteration2 18000

