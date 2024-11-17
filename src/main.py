import os
import yaml
import numpy as np
from data_preprocessing import load_dataset, split_data_by_class, augment_data
from train import train_model_on_dataset
from ensemble import ensemble_predict
from utils import plot_metrics, plot_scatter_plots
#import pandas as pd

def load_config(config_file='config/config.yaml'):
    """Load configuration settings from the YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration from the config.yaml file
    config = load_config()

    dataset_path = config['data']['path']
    augmentation_percentages = config['data']['augmentation_percentages']

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    early_stopping_patience = config['training']['early_stopping_patience']
    validation_split = config['training']['validation_split']
    
    res_plot_dir = config['results']['plots_dir']
    model_dir = config['results']['model_dir']

    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_dataset(dataset_path, validation_split)
    class_datasets = split_data_by_class(x_train, y_train)
    
    # Apply data augmentations (5%, 10%, 15% from the config)
    augmented_datasets = {}

    for aug_pct in augmentation_percentages:
        augmented_data = augment_data(class_datasets, augment_percentage=aug_pct)
        augmented_datasets[aug_pct] = augmented_data
        
        # # Save augmented data to CSV
        # for class_label, (data, labels) in augmented_data.items():
        #     # Create a DataFrame and save to CSV
        #     df = pd.DataFrame(data)
        #     df.insert(0, 'label', labels)  # Insert label as the first column
        #     df.to_csv(f'{output_dir}augmented_data_{aug_pct*100}_{class_label}.csv', index=False)
        #     print("DONE")
    

    # Train models on augmented datasets
    models = {}
    training_times = {}
    num_iterations = {}
    for aug_pct, datasets in augmented_datasets.items():
        models[aug_pct] = []
        training_times[aug_pct] = []
        num_iterations[aug_pct] = []
        for class_label, (x_data, labels) in datasets.items():
            print("Training for a class ", class_label)

            x_data = x_data.reshape(-1, 28, 28)
            
            model, training_time, history = train_model_on_dataset(x_data, labels, epochs=epochs, 
                                                                   batch_size=batch_size, 
                                                                   early_stopping_patience=early_stopping_patience,
                                                                   validation_split=validation_split)
            
            # Save models in results/models dir

            #model.save(f'{model_dir}/model_{aug_pct*100}_{class_label}.h5')
            model.save(f'{model_dir}/model_{aug_pct*100}_{class_label}.keras')

            models[aug_pct].append(model)
            training_times[aug_pct].append(training_time)
            
            plot_metrics(history, output_dir=res_plot_dir, filename=f'augmented_{aug_pct*100}_{class_label}.png')
            num_iterations[aug_pct].append(len(history['loss'])) # Length of any history param. indicates num. of epochs
        
       
    # Evaluate Ensemble models (Step 8)
    print("Evaluating Ensemble Models on Test Data:")
    ensemble_accuracies = []
    for aug_pct, models_group in models.items():
        predictions = ensemble_predict(models_group, x_test)
        accuracy = np.mean(predictions == y_test)
        ensemble_accuracies.append(accuracy)
        print(f"Ensemble Model with {aug_pct*100}% augmentation - accuracy: {accuracy:.4f}")

    times_plot = []
    num_iter_plot = []
    # Calculate the total training time for each group
    for aug_pct, times in training_times.items():
        total_time = np.sum(times)
        times_plot.append(total_time)
        print(f"Total training time for {aug_pct*100}% augmentation: {total_time:.2f} seconds")

    for aug_pct, num_iterations in num_iterations.items():
        num_iter_plot.append(np.sum(num_iterations))

    plot_scatter_plots(ensemble_accuracies, times_plot, num_iter_plot, res_plot_dir)

if __name__ == "__main__":
    main()
