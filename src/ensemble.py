import numpy as np

def ensemble_predict(models, x_data):
    """Aggregate predictions from multiple models using majority voting."""
    predictions_index = np.zeros((len(x_data), len(models)), dtype=int)  # Ensure integer type for predictions
    #predictions_prob = np.zeros((len(x_data), len(models)), dtype=float)
    
    # Get predictions from each model
    for i, model in enumerate(models):
        # Make predictions for each model - Model output is assumed to be probabilities, so we use argmax to get the predicted class

        predictions_index[:, i] = model.predict(x_data).argmax(axis=1)
        #predictions_prob[:, i] = model.predict(x_data).max(axis=1)
    
    # Perform majority voting
    final_predictions = [np.bincount(predictions_index[i]).argmax() for i in range(len(predictions_index))]
    #final_predictions_prob = [(np.argmax(predictions_prob[i])).item() for i in range(len(predictions_prob))]

    return np.array(final_predictions)
