
import numpy as np

def predict_churn(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction == 1:
        return "The customer is likely to churn."
    else:
        return "The customer is unlikely to churn."