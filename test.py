import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import initialize_model 
from config import DEVICE

def test_model(model_weights, dataset_path):
    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载训练好的模型权重
    model = initialize_model()  
    model.load_state_dict(torch.load(model_weights))
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    # 你可以根据实际情况修改路径
    model_weights_path = "E:/HuaweiMoveData/Users/Catherine/Desktop/vscode/PY/computer-vision2/model_weights/final_model.pth"
    dataset_path = "E:/HuaweiMoveData/Users/Catherine/Desktop/vscode/PY/computer-vision2/data/caltech-101"  
    test_model(model_weights_path, dataset_path)
