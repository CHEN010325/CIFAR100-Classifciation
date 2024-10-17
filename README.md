Welcome to the Elema team's codebase. This codebase is about CIFAR100 image classification


Here's a concise introduction for CIFAR-100 classification:

---

**CIFAR-100 Classification**  
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. It is split into 50,000 training and 10,000 test images. CIFAR-100 includes two levels of labels:  
- **Superclasses**: 20 high-level categories grouping the classes.
- **Fine classes**: 100 specific classes.

Common models used for CIFAR-100 classification include Convolutional Neural Networks (CNNs), ResNet, Wide-ResNet, VGGNet, and Vision Transformers (ViT). Performance metrics often focus on Top-1 and Top-5 Accuracy, and sometimes Super-Class Accuracy for broader category classification. This dataset is widely used to benchmark image recognition algorithms due to its small image size and challenging classification task with many similar classes.

---
      Here are the latest results from our team
| Random Seed | Test Acc | Top-5 Acc | Super-Class Acc |
|-------------|----------|-----------|-----------------|
| 4           | 79.05%   | 95.33%    | 87.21%          |
| 8           | 79.44%   | 95.25%    | 87.28%          |
| 12          | 79.14%   | 95.27%    | 87.74%          |
