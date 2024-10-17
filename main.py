import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F 
import numpy as np
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    绘制训练和测试的损失及准确率曲线图。

    参数:
    - train_losses: 训练集损失列表
    - test_losses: 测试集损失列表
    - train_accuracies: 训练集准确率列表
    - test_accuracies: 测试集准确率列表
    """

    # 使用更好的图表样式（如 'ggplot' 风格）
    plt.style.use('ggplot')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='--', marker='o')
    plt.plot(test_losses, label='Test Loss', color='red', linestyle='-', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.grid(True)  # 添加网格线
    plt.legend(loc='upper right')  # 图例位置
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy', color='green', linestyle='--', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange', linestyle='-', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.grid(True)  # 添加网格线
    plt.legend(loc='lower right')  # 图例位置
    plt.show()


def focal_loss(pred, target, alpha=1, gamma=2.5):

    # Convert the target into one-hot format
    target_one_hot = torch.eye(pred.size(1), device=pred.device)[target]  # Ensure one-hot is on the same device as pred

    # Compute log softmax of predictions
    log_probs = F.log_softmax(pred, dim=-1)

    # Compute probabilities
    probs = torch.exp(log_probs)

    # Focal Loss computation
    focal_weight = (1 - probs) ** gamma
    loss = -alpha * focal_weight * log_probs * target_one_hot

    return loss.sum(dim=1).mean()  # Averaging over batch


# 标签平滑损失函数实现
def label_smoothing_loss(pred, target, smoothing=0.175):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)  # 计算log概率
    true_dist = torch.zeros_like(log_probs)  # 创建与预测相同维度的tensor
    true_dist.fill_(smoothing / (pred.size(1) - 1))  # 平滑后的分布
    true_dist.scatter_(1, target.data.unsqueeze(1), confidence)  # 填充正确类的概率
    return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))  # 计算损失



import time
import torch

# CIFAR100 Super-Class mapping: 100 classes to 20 super-classes
super_class_mapping = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]  # vehicles 2
}

def train(model, trainloader, testloader, criterion, optimizer, super_class_mapping, class_to_superclass, device='cpu', num_epochs=15, patience=2, min_delta=2, use_early_stopping=True):
    """
    训练模型，并在最后计算和打印Top-1准确率、Top-5准确率、Super-Class准确率。
    """
    best_accuracy = 0.0
    no_improvement_count = 0
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_super_class_train = 0
        total_super_class_train = 0
        
        # Training phase
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # 在这里获取 labels
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 计算训练过程中的超类正确率
            for i, label in enumerate(labels):
                superclass = class_to_superclass[label.item()]
                total_super_class_train += 1
                if predicted[i].item() in super_class_mapping[superclass]:
                    correct_super_class_train += 1

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        correct_top5 = 0
        correct_super_class_test = 0
        total_super_class_test = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data  # 在这里获取 labels
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                _, top5_preds = outputs.topk(5, dim=1)
                correct_top5 += (top5_preds == labels.view(-1, 1)).sum().item()

                # 计算验证过程中的超类正确率
                for i, label in enumerate(labels):
                    superclass = class_to_superclass[label.item()]
                    total_super_class_test += 1
                    if predicted[i].item() in super_class_mapping[superclass]:
                        correct_super_class_test += 1

        test_loss = running_test_loss / len(testloader)
        test_accuracy = 100 * correct_test / total_test
        top5_accuracy = 100 * correct_top5 / total_test
        super_class_accuracy_test = 100 * correct_super_class_test / total_super_class_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print metrics
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.2f}%, '
              f'Top-5 Acc: {top5_accuracy:.2f}%, Super-Class Acc: {super_class_accuracy_test:.2f}%')

        # Early stopping check
        if use_early_stopping:
            if test_accuracy - best_accuracy <= min_delta:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            best_accuracy = max(test_accuracy, best_accuracy)

            if no_improvement_count >= patience:
                print(f'Early stopping at epoch {epoch + 1}, best accuracy: {best_accuracy:.2f}%')
                break

    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f'Finished Training in {int(minutes)}m {int(seconds)}s')

    print(f'Final Test Accuracy: {test_accuracy:.2f}%')
    print(f'Final Top-5 Accuracy: {top5_accuracy:.2f}%')
    print(f'Final Super-Class Accuracy: {super_class_accuracy_test:.2f}%')

    return train_losses, test_losses, train_accuracies, test_accuracies

# 定义 class_to_superclass 的映射
class_to_superclass = {class_idx: superclass for superclass, classes in super_class_mapping.items() for class_idx in classes}