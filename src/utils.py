import matplotlib.pyplot as plt
import os 

def plot_metrics(history, output_dir, filename):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Model Training Progress")
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_scatter_plots(accuracies, total_training_times, num_iterations, output_dir):
    """Plot Accuracy vs Training Time and Accuracy vs Number of Iterations."""
    # Plotting accuracy vs training time
    plt.figure(figsize=(12, 6))

    # Scatter plot for accuracy vs training time
    plt.subplot(1, 2, 1)
    plt.scatter(accuracies, total_training_times, color='blue')
    plt.title('Accuracy vs Training Time')
    plt.xlabel('Accuracy')
    plt.ylabel('Training Time (seconds)')
    plt.grid()

    # Scatter plot for accuracy vs number of iterations
    plt.subplot(1, 2, 2)
    plt.scatter(accuracies, num_iterations, color='green')
    plt.title('Accuracy vs Number of Iterations')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Iterations')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_training_time_and_iterations.png')
    plt.show()