## ViT vs CNNs in Low-Data Regimes
### Overview
This is a small project to investigates the behavior of Vision Transformer (ViTs) versus Convolutional Neural Networks (CNNs) when these are fine-tuned under limited labeled data conditions. Utilizing the CIFAR10 as dataset for both models, the ViT model  ```google/vit-base-patch16-224-in21k``` from the HuggingFace hub and the RestNet-18 CNN model from PyTorch.  

## Research Question
How do pretrained Vision Transformers compare to CNNs when trained with limeted labeled data?
## Experimental Setup
### Dataset
- Dataset: CIFAR-10
- Number of classes: 10
- Image resolution: 32x32 (resized to 224x224 for ViT)

| Split                | Samples |
| -------------------- | ------- |
| Training (ViT)       | 1,000   |
| Training (ResNet-18) | 50,000  |
| Validation/Test      | 10,000  |

> Training set sizes differ between models. This is discussed in the analysis.

## Models
### Vision Transformer (ViT)
- Architecture: ```google/vit-base-patch16-224-in21k```
- Pretraining: ImageNet-21k
- Framework: Hugging Face Transformers
- Input size: 224×224
- Fine-tuning strategy: Full model HuggingFace fine-tuning

### Convolutional Neural Network (CNN)
- Architecture: ResNet-18
- Initialization: ImageNet pretrained weights
- Framework: PyTorch / torchvision
- Input size: 32×32
- Training strategy: End-to-end supervised training

## Training Details
- Optimizer: Adam
- Learning rate: 2e-4
- Epochs: 10
- Loss Function: Cross-Entropy Loss
- Hardware: NVIDIA GeForce GTX 1660 D:

## Model Metrics
### Vision Transformer (ViT)
| Epoch | Train Loss | Val Loss | Accuracy |
| ----- | ---------- | -------- | -------- |
| 1     | —          | 0.3559   | 93.50%   |
| 2     | 0.3683     | 0.3236   | 92.27%   |
| 3     | 0.3683     | 0.2506   | 93.84%   |
| 4     | 0.1117     | 0.2568   | 93.58%   |
| 5     | 0.0416     | 0.2468   | 93.76%   |
| 6     | 0.0416     | 0.2478   | 93.69%   |
| 7     | 0.0271     | 0.2496   | 93.64%   |
| 8     | 0.0213     | 0.2520   | 93.59%   |
| 9     | 0.0213     | 0.2531   | 93.60%   |
| 10    | 0.0187     | 0.2535   | 93.60%   |

### ResNet-18 (CNN)
| Epoch | Train Loss | Val Loss | Accuracy |
| ----- | ---------- | -------- | ------------ |
| 1     | 0.4147     | 0.4654   | 84.10%       |
| 2     | 0.4037     | 0.4630   | 84.62%       |
| 3     | 0.3833     | 0.4478   | 84.68%       |
| 4     | 0.3736     | 0.4510   | 84.94%       |
| 5     | 0.3623     | 0.4519   | 84.99%       |
| 6     | 0.3454     | 0.4669   | 84.97%       |
| 7     | 0.3335     | 0.4439   | 85.29%       |
| 8     | 0.3236     | 0.4342   | 86.09%       |
| 9     | 0.3182     | 0.4394   | 85.78%       |
| 10    | 0.3058     | 0.4348   | 85.86%       |

## Observations
| Model          | Final Train Loss | Final Val Loss | Final Accuracy | Training Time |
| ---------------| ---------------- | -------------- | -------------- | ------------- |
| ViT-B/16       |    ~0.019        |         ~0.254 |         93.60% |      ~73 mins |
| RestNet-18 CNN |          ~0.306  |         ~0.435 |         85.86% |       ~6 mins |
1. In the ViT, accuracy is already high in epoch 1, this is a big signal of how pretrained models helps for fine-tuning in a small data regimes, even with far less samples.
2. The CNN model improves train loss, val loss and accuracy gradually and stays up with 85.86% of accuracy, with yes the whole dataset, but with significant fraction of time compared to the ViT model (I hypothesize that ViT time was high due to the preprocessing of images, going from 32x32 to 224x224).
3. The ViT model shows signs of rapid learning from 0.368 training loss to 0.0271 already at epoch 7. Vaidation loss stops improving after epoch 2, staying the same for all the next epochs. Meaning there is a mild overfitting, since there is no improvement.
4. While the ViT required approximately 70 minutes of training, the ResNet-18 achieved high accuracy within 6–7 minutes on the same hardware.
## Analysis
The comparison between a ViT trained on a small subset (1,000 samples) and a CNN trained on the full dataset was not the original experimental objective, but rather emerged during the implementation process. 
Despite this, the results highlighted the strong sample efficiency of a large-scale pretrained Vision Transformer, which achieved competitive performance using only a fraction of the available training data.
At the same time, this experiment exposed the significant computational cost associated with fine-tuning large pretrained transformers.
This highlights an important trade-off: although pretrained ViTs can be highly sample-efficient, CNNs such as ResNet-18 remain attractive in practice due to their computational efficiency and faster convergence.
## Conclusion
Giving the observation of this experiment, heavily pretrained on large-scale datasets, Vision Transformers exhibit strong sample efficiency and can generalize well with very limited labeled data, often outperforming smaller CNNs such as ResNet-18 under the same fine-tuning conditions. However, In contrast, the ResNet-18 model, trained on the full dataset, converged substantially faster and reached competitive accuracy within a fraction of the training time. This contrast illustrates a trade-off between sample efficiency and computational efficiency, emphasizing that CNNs remain a strong practical choice in scenarios with limited computational resources.

## Reproducibility
If you want to reproduce this experiment simply copy this repo or download it! Just make sure to have:
- python 3.x

Install all dependencies:
```
pip install -r requiremets.txt
```
and check jupyter notebooks! :D

