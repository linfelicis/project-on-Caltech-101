import os
import shutil
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def prepare_dataset():
    
    # 使用Path处理路径更安全
    base_dir = Path(r"E:\HuaweiMoveData\Users\Catherine\Desktop\vscode\PY\computer-vision2\data\caltech-101")
    src_dir = base_dir / "101_ObjectCategories"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"

    # 验证原始数据
    if not src_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {src_dir}")

    # 自动划分数据集（如果尚未划分）
    if not train_dir.exists():
        print("正在自动划分数据集...")
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        for class_name in os.listdir(src_dir):
            class_path = src_dir / class_name
            if class_path.is_dir() and class_name != "BACKGROUND_Google":  # 跳过背景类
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(images) < 40:
                    print(f"警告：类别 {class_name} 只有 {len(images)} 张图片")
                    continue
                
                np.random.shuffle(images)
                
                # 创建目标目录
                (train_dir / class_name).mkdir(exist_ok=True)
                (test_dir / class_name).mkdir(exist_ok=True)

                # 复制文件
                for i, img in enumerate(images):
                    src = class_path / img
                    dst = (train_dir if i < 30 else test_dir) / class_name / img
                    shutil.copy(src, dst)
        
        print(f"划分完成：\n"
              f"- 训练集: {len(list(train_dir.glob('*')))} 类\n"
              f"- 测试集: {len(list(test_dir.glob('*')))} 类")

def get_data_loaders():
    
    prepare_dataset()  # 确保数据已准备
    
    base_dir = Path(r"E:\HuaweiMoveData\Users\Catherine\Desktop\vscode\PY\computer-vision2\data\caltech-101")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = datasets.ImageFolder(
        str(base_dir / "train"),
        transform
    )
    test_dataset = datasets.ImageFolder(
        str(base_dir / "test"), 
        transform
    )

    return (
        {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
            'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
        },
        {
            'train': len(train_dataset),
            'test': len(test_dataset)
        },
        train_dataset.classes
    )