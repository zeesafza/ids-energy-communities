import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data
normal = pd.read_csv("data/normal/features_dae.csv")

# Convert complex to magnitude
def convert_to_magnitude(value):
    if isinstance(value, complex):
        return abs(value)
    elif isinstance(value, str):
        value = value.replace('i', 'j')
        try:
            return abs(complex(value))
        except ValueError:
            return value
    else:
        return value

for col in normal.columns:
    normal[col] = normal[col].apply(convert_to_magnitude)

# Split the data into num_clients parts
num_clients = 3  # Number of clients
client_data = np.array_split(normal, num_clients)

# Prepare training data for this client
def prepare_data(client_id):
    df_small_noise = client_data[client_id].iloc[:, 1:]  # Remove time column
    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std
    return df_training_value

# Create sequences
TIME_STEPS = 10

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)

# Define the model
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        layers.LSTM(64, return_sequences=False),
        layers.RepeatVector(TIME_STEPS),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(128, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train):
        self.model = model
        self.x_train = x_train

    def get_parameters(self, config):
        logging.debug("get_parameters called")
        return self.model.get_weights()

    def fit(self, parameters, config):
        logging.debug("fit called with 50 epochs")
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.x_train, epochs=50, batch_size=128, verbose=0)
        logging.debug("fit completed")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        logging.debug("evaluate called")
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.x_train, self.x_train, verbose=0)
        logging.debug(f"evaluate completed with loss={loss}")
        
        if config.get("final_round", False):
            self.model.save('federated-model.keras', overwrite=True)
            logging.debug("Final model saved as federated_final_model.keras")

        
        return float(loss), len(self.x_train), {"loss": float(loss)}

# Start the client
def start_client(client_id):
    # Prepare data for this client
    df_training_value = prepare_data(client_id)
    x_train = create_sequences(df_training_value)
    x_train = np.array(x_train, dtype=np.float32)

    # Create the model
    model = create_model((TIME_STEPS, df_training_value.shape[1]))

    # Start the Flower client
    client = FlowerClient(model, x_train)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])  # Pass client_id as a command-line argument
    start_client(client_id)
