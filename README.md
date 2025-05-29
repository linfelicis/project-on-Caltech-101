本项目使用 **PyTorch** 框架实现了基于 **ResNet-18** 的 **Caltech-101** 分类任务。代码可以在 **Caltech-101** 数据集上训练模型，并进行微调。

## 任务要求

- 微调 **ResNet-18** 模型，使其适应 **Caltech-101** 数据集。
- 在 **训练** 和 **验证** 过程中，通过 **TensorBoard** 可视化 **loss 曲线** 和 **accuracy 变化**。
- 通过对比 **使用预训练模型** 和 **从零开始训练** 的效果，展示预训练模型带来的提升。

## 环境要求

请确保安装以下依赖项：
- Python 3.6+
- PyTorch 1.8+
- torchvision
- TensorBoard
- matplotlib
- scikit-learn

数据集准备
下载 Caltech-101 数据集：

数据集可以从 Caltech-101 下载。

解压数据集并放入项目目录中的 ./data 文件夹。

数据集结构应如下所示：
```
./data/
├── caltech101/
    ├── plane/
    ├── car/
    ├── dog/
    ├── ...
```
## 使用说明
# 1. 数据加载
数据加载器会将 Caltech-101 数据集划分为训练集和测试集，并进行预处理。你可以在 data_loader.py 中找到数据加载和预处理的代码。

# 2. 模型初始化与训练
要训练模型，首先确保你已经准备好数据集，然后运行 main.py 脚本：
```
python main.py
```
在脚本中，模型将根据以下配置进行训练：

模型架构：ResNet-18

训练轮数：15轮（你可以根据需要调整）

学习率：0.001

微调策略：冻结卷积层，仅训练输出层

# 3. TensorBoard 可视化
运行模型后，你可以使用 TensorBoard 来可视化训练过程中的 loss 和 accuracy。

启动 TensorBoard：
```
tensorboard --logdir=runs
```
然后，打开浏览器并访问 http://localhost:6006 来查看训练过程中的可视化效果。

# 4. 保存与加载模型权重
训练完成后，模型会自动保存 最佳权重。你可以在 ./model_weights 目录下找到 best_model.pth 和 final_best_model.pth 文件。

要加载已保存的模型权重，使用以下代码：
```
# 加载模型权重
model.load_state_dict(torch.load('./model_weights/best_model.pth'))
model.eval()  # 切换到评估模式
```
## 训练与评估
# 训练过程
训练过程中，模型会打印 训练损失 和 训练准确率，同时记录 验证集的损失 和 验证准确率。
模型训练会使用 学习率衰减，每 30 个 epoch 学习率衰减为原来的 0.1。

# 模型评估
训练过程中，你可以查看 训练集 和 验证集 上的 accuracy 曲线，并进行比较：

Top-1 Accuracy：在验证集上模型正确分类的样本占总样本的比例。

Top-5 Accuracy：验证集上，模型预测的前 5 个类别中是否包含真实标签。

# 项目结构
```
├── main.py           # 训练脚本
├── model.py          # 模型初始化和设置
├── data_loader.py    # 数据集加载和预处理
├── config.py         # 配置文件，设置设备、学习率等
└── model_weights/    # 保存模型权重
```
