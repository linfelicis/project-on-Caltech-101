import torch
from torch.utils.tensorboard import SummaryWriter
from config import DEVICE
import torchvision.transforms as transforms
from PIL import Image


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):

    best_acc = 0.0
    
    # 创建 TensorBoard 日志记录器
    log_dir = "./runs/exp1"  # 日志目录
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为验证模式
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到 TensorBoard
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                # 记录训练图像（选择一批图像并记录）
                writer.add_images('Train Images', inputs, epoch)
            elif phase == 'test':
                writer.add_scalar('Loss/test', epoch_loss, epoch)
                writer.add_scalar('Accuracy/test', epoch_acc, epoch)
                # 记录测试图像
                writer.add_images('Test Images', inputs, epoch)
            
            # 更新最佳验证准确率
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
        
        print()
    
    print(f'Best test Acc: {best_acc:.4f}')
    
    # 关闭 TensorBoard 写入器
    writer.close()
    
    return model, best_acc