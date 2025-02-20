import numpy as np
import pandas as pd
import keras
import seaborn as sns
from keras import layers
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load Normal Data
normal = pd.read_csv("data/normal/features_dae.csv")

# List of Attack Datasets
  
attack_files = {
    "PA x2": "data/attack/attack/features_attack_BATTERYI-2dae.csv",
    #"PA x5": "data/attack/attack/features_attack_BATTERYI-5dae.csv",
    #"PA x10": "data/attack/attack/features_attack_BATTERYI-10dae.csv",
    
    #"DoS": "data/attack/attack/features_attack_BATTERYI-0dae.csv",
    
    #"PI-1": "data/attack/attack/features_attack_BATTERYI-minus1dae.csv",
    #"PI-2": "data/attack/attack/features_attack_BATTERYI-minus2dae.csv",
    #"PI-5": "data/attack/attack/features_attack_BATTERYI-minus5dae.csv",
 
    
    #"LR x0": "data/attack/attack/features_attack_LOADI-0.csv",
    #"LR x0.5": "data/attack/attack/features_attack_LOADI-0.5.csv",
        
    #"LI x2": "data/attack/attack/features_attack_LOADI-2.csv",
    #"LI x5": "data/attack/attack/features_attack_LOADI-5.csv",
    
 
    
}

# Convert Complex to Magnitude
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

# Apply Conversion to Normal Data
for col in normal.columns:
    normal[col] = normal[col].apply(convert_to_magnitude)

# Prepare Training Data
df_small_noise = normal.iloc[:, 1:]  # Remove time column
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std

TIME_STEPS = 10

# Function to Create Sequences
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)

x_train = create_sequences(df_training_value)
x_train = np.array(x_train, dtype=np.float32)

# Define Autoencoder Model
model = keras.Sequential([
    layers.LSTM(128, input_shape=(TIME_STEPS, df_training_value.shape[1]), return_sequences=True),
    layers.LSTM(64, return_sequences=False),
    layers.RepeatVector(TIME_STEPS),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(df_training_value.shape[1]))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train Model
history = model.fit(
    x_train, x_train, epochs=50, batch_size=128, validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")]
)

# Save Model
#model.save('detection-model.keras', overwrite=True)

# Load Saved Model
FL = "0"

if FL == "1":
  print("Using Federated model")
  model = keras.saving.load_model('federated-model.keras')
 
  #remove below for centralized model
  history = model.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")])
else:
  print("Using Centralized model")
  model = keras.saving.load_model('detection-model.keras')

#Plot
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Training Loss")
#plt.plot(history['val_loss'], label='Validation Loss', color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.title("Model Training Performance")
#plt.legend()
#plt.grid()
plt.savefig('evaluation/FDI/training-performance.svg')
plt.show()

# Compute Training Loss Distribution
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
threshold = np.percentile(train_mae_loss, 95)  # 95th percentile as threshold

# Prepare Detection Performance Plot
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y']
attack_labels = []
all_losses = {}

# Evaluate Each Attack
for i, (attack_name, attack_path) in enumerate(attack_files.items()):
    attack = pd.read_csv(attack_path)

    # Convert to Real Numbers
    for col in attack.columns:
        attack[col] = attack[col].apply(convert_to_magnitude)

    # Prepare Test Data
    df_test_value = attack.iloc[:, 1:]  # Remove time column
    df_test_value = (df_test_value - training_mean) / training_std
    x_test = create_sequences(df_test_value)
    x_test = np.array(x_test, dtype=np.float32)

    # Get Test MAE Loss
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=(1, 2))
    test_mae_loss = test_mae_loss.reshape((-1))

    # Store losses for later visualization
    all_losses[attack_name] = test_mae_loss

    # Plot Detection Performance for this Attack
    plt.plot(test_mae_loss, color=colors[i], label=attack_name)

    

# Add Threshold Line

plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, label=f"T = {threshold:.3f}")

# Legend, Labels & Save
plt.xlabel("Sample Index", labelpad=20)
plt.tick_params(axis='x', pad=20) 
plt.ylabel("Reconstruction Error", labelpad=20)
plt.legend(loc='upper left', frameon=False, labelspacing=1.2)
plt.savefig('evaluation/FDI/detection-performance.svg')
plt.show()

# ----------------------------------------------
# Distribution of Reconstruction Error for Each Attack
# ----------------------------------------------

plt.figure(figsize=(12, 6))
for i, (attack_name, loss_values) in enumerate(all_losses.items()):
    #kde = sns.kdeplot(loss_values, color=colors[i], label=attack_name, fill=False)
    sns.kdeplot(loss_values, color=colors[i], label=attack_name, fill=True, alpha=0.5)

    # Extract X (Reconstruction Error) and Y (Density) values
    #x, y = kde.get_lines()[-1].get_data()

    # Fill Before Threshold
    #plt.fill_between(x, y, where=(x <= threshold), color='black', alpha=0.3)

    # Fill After Threshold
    #plt.fill_between(x, y, where=(x > threshold), color=colors[i], alpha=0.3)

# Add Threshold Line
plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, label=f"T = {threshold:.3f}")

# Labels, Title, and Save
plt.xlabel("Reconstruction Error", labelpad=25)
plt.tick_params(axis='x', pad=20) 
plt.ylabel("Density", labelpad=20)
plt.legend(loc='upper left', frameon=False, labelspacing=1.2)
#plt.title("Distribution of Reconstruction Error for Each Attack")
plt.savefig('evaluation/FDI/reconstruction-error-distribution.svg')
plt.show()

# ----------------------------------------------
# Confusion Matrix for Each Attack
# ----------------------------------------------
for i, (attack_name, attack_path) in enumerate(attack_files.items()):
    attack = pd.read_csv(attack_path)

    # Convert to Real Numbers
    for col in attack.columns:
        attack[col] = attack[col].apply(convert_to_magnitude)

    # Prepare Test Data
    df_test_value = attack.iloc[:, 1:]  # Remove time column
    df_test_value = (df_test_value - training_mean) / training_std
    x_test = create_sequences(df_test_value)
    x_test = np.array(x_test, dtype=np.float32)

    # Get Test MAE Loss
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=(1, 2))
    test_mae_loss = test_mae_loss.reshape((-1))

    # Binary Classification
    y_true = np.zeros(len(x_test))
    y_true[num_normal_samples_start: len(x_test) - num_normal_samples_end] = 1
    #y_true = np.ones(len(x_test))  # Attack samples (all 1s)
    
    y_pred = (test_mae_loss > threshold).astype(int)  # Predicted labels
    
    # Evaluation Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAttack: {attack_name}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    attack_labels.append(attack_name)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nConfusion Matrix for {attack_name}:")
    print(cm)
    

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {attack_name}")

    # Save Confusion Matrix Plot
    plt.savefig(f'evaluation/FDI/confusion_matrix_{attack_name}.svg')
    plt.close()
