import torch.optim as optim
import torch.nn as nn
import torch
import os
from model import initialize_model
from config import *
from data_loader import get_data_loaders
from train_utils import train_model

def save_best_model(model, accuracy, best_accuracy, output_dir, filename="best_model.pth"):
    """保存最佳模型的函数，基于验证集准确率"""
    if accuracy > best_accuracy:
        os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
        model_path = os.path.join(output_dir, filename)
        torch.save(model.state_dict(), model_path)  # 保存模型权重
        print(f"Best model saved with accuracy: {accuracy:.4f} at {model_path}")
        return accuracy  # 更新最佳准确率
    return best_accuracy

def setup_optimizer_and_criterion(model):
    """设置优化器和损失函数"""
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},  # 冻结的预训练部分的较小学习率
        {'params': model.fc.parameters(), 'lr': 1e-3}      # 新的输出层较大的学习率
    ], lr=LEARNING_RATE)  # 总学习率
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def train_and_validate(model, dataloaders, dataset_sizes, criterion, optimizer):
    """训练和验证模型"""
    best_acc = 0.0  # 记录最佳准确率
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        # 训练模型
        model.train()  # 进入训练模式
        train_loss, train_acc = train_model(model, dataloaders,dataset_sizes, criterion, optimizer,NUM_EPOCHS)
        
        # 验证模型
        model.eval()  # 进入评估模式
        val_loss, val_acc = validate_model(model, dataloaders['val'], criterion)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_best_model(model, './model_weights', filename="best_model.pth")
        
        # 学习率衰减（如果需要）
        if (epoch + 1) % 30 == 0:
            adjust_learning_rate(optimizer, 0.1)  # 每30个epoch学习率衰减为原来的0.1

    return best_acc

def adjust_learning_rate(optimizer, factor):
    """调整学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor

def validate_model(model, dataloader, criterion):
    """验证模型"""
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # 1. 自动准备数据（包含自动划分）   
    dataloaders, dataset_sizes, class_names = get_data_loaders()
    print(f"训练集大小: {dataset_sizes['train']}, 测试集大小: {dataset_sizes['test']}")
    
    # 2. 初始化模型，设置 feature_extract 为 True，表示只训练最后一层
    model = initialize_model()

    # 3. 设置优化器和损失函数
    optimizer, criterion = setup_optimizer_and_criterion(model)
    
    # 4. 训练和验证
    best_acc = train_and_validate(model, dataloaders, dataset_sizes, criterion, optimizer)
    
    print(f"\n训练完成，最佳测试准确率: {best_acc:.4f}")

     # 训练完成后保存模型
    save_best_model(model, './model_weights', filename='final_best_model.pth')

if __name__ == '__main__':
    main()
