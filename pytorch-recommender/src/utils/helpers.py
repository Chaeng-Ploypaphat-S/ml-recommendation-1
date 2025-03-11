def normalize_data(data):
    return (data - data.mean()) / data.std()

def calculate_metrics(predictions, targets):
    mse = ((predictions - targets) ** 2).mean()
    return mse

def log_message(message):
    print(f"[LOG] {message}")