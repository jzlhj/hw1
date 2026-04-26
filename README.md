# HW1：基于 NumPy 从零实现 EuroSAT 三层 MLP 分类器

本项目为深度学习课程 HW1，实现了一个不依赖 PyTorch / TensorFlow / JAX 的三层 MLP，用于 EuroSAT RGB 遥感图像分类。代码中使用 NumPy 自主完成前向传播、自动微分、反向传播、SGD、学习率衰减、交叉熵损失和 L2 正则化。

当前效果最好的实验结果目录为：

- `tune_relu_224_112_lr0015`

该结果对应：

- 最佳验证集准确率：`0.6472`
- 测试集准确率：`0.6378`

## 项目结构

- `src/data.py`：数据加载、分层切分、标准化、DataLoader
- `src/autograd.py`：自动微分与反向传播
- `src/model.py`：三层 MLP 模型定义
- `src/train_utils.py`：训练流程、SGD、损失函数、模型保存
- `src/eval_utils.py`：测试评估与混淆矩阵
- `src/viz.py`：训练曲线、权重图、错例图可视化
- `train.py`：训练脚本
- `test.py`：测试脚本
- `visualize.py`：生成图像可视化结果
- `search.py`：超参数搜索脚本
- `REPORT.md`：实验报告 Markdown 版本

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集说明

数据集目录应位于：

```text
EuroSAT_RGB/
```

该目录已经按类别分好子文件夹，可直接用于训练。

## 训练最终推荐模型

下面这组参数是目前实验中效果最好的配置：

```bash
python train.py --data_root EuroSAT_RGB --out_dir tune_relu_224_112_lr0015 --epochs 18 --batch_size 64 --lr 0.015 --hidden1 224 --hidden2 112 --activation relu --weight_decay 0.0001
```

训练完成后会在输出目录中生成：

- `best_model.npz`
- `history.json`
- `training_curves.png`
- `classes.json`
- `splits.json`
- `norm_stats.npz`

## 测试模型

```bash
python test.py --out_dir tune_relu_224_112_lr0015 --batch_size 64
```

测试完成后会生成：

- `test_metrics.json`
- `confusion_matrix.npy`
- `confusion_matrix.png`
- `errors.json`

## 生成可视化结果

```bash
python visualize.py --out_dir tune_relu_224_112_lr0015 --num_weights 16 --num_errors 12
```

可视化结果包括：

- `first_layer_weights.png`
- `error_examples.png`

## 超参数搜索

网格搜索：

```bash
python search.py --data_root EuroSAT_RGB --out_dir outputs_search --mode grid --epochs 6
```

随机搜索：

```bash
python search.py --data_root EuroSAT_RGB --out_dir outputs_search --mode random --trials 8 --epochs 6
```

搜索结果会保存到：

- `outputs_search/search_results.json`

## 目录说明

目前目录里会看到多个输出文件夹，它们的含义如下：

- `outputs`：一组稳定但不是最优的训练结果
- `tune_relu_*`：调参阶段产生的多组实验结果
- `tune_relu_224_112_lr0015`：当前推荐提交使用的最佳结果


## 模型权重下载

如果提交时需要将模型权重上传到网盘，请在这里补充下载地址：

- 下载链接：[`TODO`](https://drive.google.com/file/d/1IqamaLg3IYoW295HQ7ZP8d8MNo8WXxri/view?usp=drive_link)





