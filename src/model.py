from tensorflow.keras import layers, models

def create_feedforward_nn(input_shape=(28, 28, 1)):
    """Create a feedforward neural network with 3 layers, each with 50 neurons."""
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 output neurons for 10 classes
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
