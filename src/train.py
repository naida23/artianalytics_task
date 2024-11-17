from tensorflow.keras.callbacks import EarlyStopping
from model import create_feedforward_nn
import time

def train_model_on_dataset(x_train_data, y_train_data, epochs=50, batch_size=32, early_stopping_patience=5, validation_split = 0.2):
    """Train the model on the provided dataset with early stopping."""
    model = create_feedforward_nn()
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    
    # Timing the training process
    start_time = time.time()
    
    # Train the model with validation split
    history = model.fit(x_train_data, y_train_data, validation_split=validation_split, epochs=epochs, batch_size=batch_size, 
                        callbacks=[early_stopping], verbose=1)
    
    # Calculate total training time
    training_time = time.time() - start_time
    return model, training_time, history.history
