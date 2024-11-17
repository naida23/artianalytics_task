# ArtiAnalytics AI Engineer Assignment

This project implements the steps given in AI Engineer Assignment by ArtiAnalytics.
The task is to develop a simple feedforward neural network and observe how it performs on given mnist data.
Dataset should be split into 10 subsets and then augmented as explained in the assignment sheet.

# Repository Structure
The repository is organized as follows:
```config/ # Configuration files │ ├── config.yaml │ data/ # Original and augmented datasets │ ├── raw/ # Original dataset │ └── processed/ # Augmented datasets │ results/ # Directory to save plots and models │ ├── plots/ # Saved plots │ └── models/ # Saved models │ src/ # Source code │ ├── main.py # Main script to run the project ├── utils.py # Utility functions ├── train.py # Training script ├── ensemble.py # Ensemble methods └── data_preprocessing.py # Data preprocessing script │ ARTI_Analytics_Task_AI_Eng.pdf # Project report README.md # Project documentation requirements.txt # Python dependencies```


# How to run the application

To set up the project, you need to install the required dependencies listed in the requirements.txt file. You can do this by running the following command in your terminal:
`pip install -r requirements.txt`

After installing the dependencies, you can run the main script of the application by executing:
`python src/main.py`

# Data Preprocessing
The dataset is divided into 10 subsets based on their labels. Each of these 10 subsets is then augmented by adding 5%, 10%, and 15% of randomly selected samples from the other 9 subsets.

# Training
A simple feedforward neural network with 3 layers, each with 50 neurons, has been created. Parameters used to train the network:
| Parameter                | Value                |
|--------------------------|----------------------|
| Activation Function      | ReLU, SoftMax        |
| Optimizer                | Adam                 |
| Learning Rate            | 0.001 (configurable) |
| Epochs                   | 50 (configurable)    |
| Batch Size               | 32 (configurable)    |
| Early-Stopping Patience  | 5 (configurable)     |

Each subset is split into training/testing dataset with 80/20 ratio.

- Total training time for 5.0% augmentation: 52.22 seconds,
- Total training time for 10.0% augmentation: 53.78 seconds,
- Total training time for 15.0% augmentation: 53.26 seconds.


# Results


|     Class 0         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.0273e-05        |        76.23%          |         3.70          | 
|  Dataset + 10% Aug. |       100%          |   2.7764e-05        |        54.64%          |         7.50          | 
|  Dataset + 15% Aug. |       100%          |   7.1437e-07        |        34.80%          |        13.44          |


|     Class 1         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.3669e-06        |        76.21%          |         7.98          | 
|  Dataset + 10% Aug. |       100%          |   6.5727e-06        |        54.57%          |         14.56         | 
|  Dataset + 15% Aug. |       100%          |   2.3674e-05        |        34.80%          |         19.72         |


|     Class 2         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.7006e-06        |        76.22%          |        6.0097         | 
|  Dataset + 10% Aug. |       100%          |   3.7377e-05        |        54.61%          |         8.57          | 
|  Dataset + 15% Aug. |       100%          |   6.7143e-07        |        34.88%          |        18.32          |


|     Class 3         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   2.1439e-05        |        76.26%          |         4.71          | 
|  Dataset + 10% Aug. |       100%          |   1.4417e-06        |        54.58%          |         10.73         | 
|  Dataset + 15% Aug. |       100%          |   1.5562e-05        |        34.87%          |         13.98         |


|     Class 4         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   3.7124e-05        |        76.22%          |         4.72          | 
|  Dataset + 10% Aug. |       100%          |   1.0967e-05        |        54.63%          |         10.54         | 
|  Dataset + 15% Aug. |       100%          |   9.5857e-06        |        34.86%          |         15.12         |


|     Class 5         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   3.1523e-05        |        76.30%          |         5.07          | 
|  Dataset + 10% Aug. |       100%          |   1.9260e-05        |        54.63%          |         10.20         | 
|  Dataset + 15% Aug. |       100%          |   2.9284e-05        |        34.84%          |         13.92         |


|     Class 6         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.0555e-06        |        76.20%          |         5.71          | 
|  Dataset + 10% Aug. |       100%          |   9.6999e-06        |        54.55%          |         8.304         | 
|  Dataset + 15% Aug. |       100%          |   6.4048e-06        |        34.83%          |         13.34         |


|     Class 7         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   2.3187e-05        |        76.22%          |         5.13          | 
|  Dataset + 10% Aug. |       100%          |   9.9333e-06        |        54.62%          |         10.42         | 
|  Dataset + 15% Aug. |       100%          |   7.0631e-06        |        34.85%          |         15.93         |


|     Class 8         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.3975e-06        |        76.24%          |         4.95          | 
|  Dataset + 10% Aug. |       100%          |   1.0286e-05        |        54.55%          |         8.15          | 
|  Dataset + 15% Aug. |       100%          |   1.1634e-06        |        34.85%          |         14.5         |


|     Class 9         | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.0447e-05        |        76.28%          |         4.96          | 
|  Dataset + 10% Aug. |       100%          |   3.6537e-06        |        54.59%          |         10.87         | 
|  Dataset + 15% Aug. |       100%          |   1.8970e-05        |        34.83%          |         13.39         |


|     Avg             | Training Acc.       | Training Loss       |  Validation Acc.       | Validation Loss       |
|---------------------|---------------------|---------------------|------------------------|-----------------------|
|  Dataset + 5% Aug.  |       100%          |   1.39e-05          |        76.23%          |         5.29          | 
|  Dataset + 10% Aug. |       100%          |   1.47e-05          |        54.59%          |         9.98          | 
|  Dataset + 15% Aug. |       100%          |   2.118e-05         |        34.84%          |         15.17         |

# Interpretation of Results
We can observe that all the models overfit, which is expected given the severe class imbalance in the dataset. The model achieves high training accuracy by learning to predict the dominant class for most (or all) of the data. This happens because predicting the dominant class for the majority of inputs results in a low training loss. Since the validation loss and accuracy do not improve initially, early stopping is triggered, and training halts after 6 epochs (when early stopping patience is set to 5).

Increasing the number of samples from other datasets leads to a decrease in validation accuracy. This is because the model now learns to generalize better across other classes and becomes less biased towards the dominant class. While the validation accuracy is still computed on a subset where the dominant class is overrepresented, we can see that the model no longer predicts the major class for all inputs, as it did when the training dataset was less augmented. However, 15% augmentation is still insufficient for the model to generalize well.

 # Majority Voting

Since each of the 10 models is overfitting and thus predicting the class it was trained on, the resulting predictions are always the same 10-element array: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, where each element represents the prediction of a different model. This means that every class label is predicted exactly once, so the voting process will always select the first class as the final prediction. As a result, the accuracy remains constant at 9.79%, regardless of how much the dataset is augmented, since 15% is still too small a subset to enable the models to generalize better.