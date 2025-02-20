import flwr as fl
from flwr.server import ServerConfig

def start_server(num_rounds=3):
    # Define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        min_fit_clients=3,  # Minimum number of clients to be sampled for training
        min_available_clients=3,  # Minimum number of available clients
        on_evaluate_config_fn=lambda rnd: {"final_round": rnd == num_rounds},  # Send final_round flag
    )

    # Define the server configuration
    config = ServerConfig(num_rounds=num_rounds)

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Use IPv4 address
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
