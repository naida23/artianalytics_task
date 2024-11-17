import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def load_dataset(path, test_size):
    """Load dataset."""

    # Load the CSV file directly into pandas
    df = pd.read_csv(path)

    print("Dataset loaded!") 
    
    # Separate the labels and the features (pixels)
    labels = df.iloc[:, 0].values  # The first column contains the labels
    features = df.iloc[:, 1:].values  # The remaining columns are the pixel values

    # Reshape the features into 28x28 images
    # The features are flattened, so we need to reshape them into 28x28 arrays
    images = features.reshape(-1, 28, 28)

    # Normalize the pixel values (scale to [0, 1])
    # Use MinMaxScaler to normalize pixel values between 0 and 1
    scaler = MinMaxScaler()
    images = scaler.fit_transform(images.reshape(-1, 28 * 28))  # Flatten the 28x28 images before scaling
    images = images.reshape(-1, 28, 28)  # Reshape back to 28x28

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test


def split_data_by_class(x_train, y_train):
    """Split the training data into 10 class-specific datasets."""
    class_datasets = {i: x_train[y_train == i] for i in range(10)}
    return class_datasets

def augment_data(class_datasets, augment_percentage=0.05):
    """Augment a given class dataset by adding data from other classes."""
    augmented_datasets = {}
    for class_label, class_data in class_datasets.items():
        # Calculate the number of additional samples
        augment_size = int(len(class_data) * augment_percentage)
        other_classes = [i for i in range(10) if i != class_label]
        
        # Randomly pick data from other classes
        all_other_data = np.concatenate(
            [np.concatenate((class_datasets[selected_class].reshape(class_datasets[selected_class].shape[0], -1), 
                             np.full((len(class_datasets[selected_class]), 1), selected_class)), axis=1) for selected_class in other_classes], 
            axis=0
        )
        
        rows = np.random.choice(np.arange((all_other_data.shape[0])), size=augment_size, replace=False)  # Randomly select samples
        new_data = all_other_data[rows]
        
        augmented_data = np.concatenate([class_data.reshape((class_data.shape[0],-1)), new_data[:, :-1]], axis=0)  # Keep only the image data
        augmented_labels = np.concatenate([np.full(len(class_data), class_label), new_data[:, -1]], axis=0)  # Create labels for augmented data
        augmented_datasets[class_label] = (augmented_data, augmented_labels)  # Store images and labels separately
    
    return augmented_datasets
