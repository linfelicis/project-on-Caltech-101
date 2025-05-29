import torch.nn as nn
from torchvision import models
from config import MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, DEVICE

def initialize_model():
    """
    初始化预训练模型并修改最后一层
    返回:
        model (nn.Module): 初始化后的模型
    """
    # 加载预训练模型
    model = models.__dict__[MODEL_NAME](pretrained=False)
    
    # 冻结所有参数
    if FEATURE_EXTRACT:
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改最后一层
    if hasattr(model, 'fc'):  # ResNet系列
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif hasattr(model, 'classifier'):  # AlexNet/VGG
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model.to(DEVICE)