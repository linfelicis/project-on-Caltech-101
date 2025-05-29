import torch

DATA_DIR = "E:/HuaweiMoveData/Users/Catherine/Desktop/vscode/PY/computer-vision2/data/caltech-101"  # 数据集路径
BATCH_SIZE = 1024               # 批量大小
IMAGE_SIZE = 224                # 图像尺寸

# 训练配置
FEATURE_EXTRACT = True         # 是否冻结特征提取层
NUM_EPOCHS = 15                # 训练轮数
LEARNING_RATE = 0.01          # 初始学习率
MOMENTUM = 0.9                 # 优化器动量

# 模型配置
MODEL_NAME = 'resnet18'        # 使用的模型架构
NUM_CLASSES = 101              # 分类类别数

# 设备配置
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'