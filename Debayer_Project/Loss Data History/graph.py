import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('loss_v10.csv')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['Train_Loss'], label='Train Loss')
plt.plot(df['Test_Loss'], label='Test Loss')

# Add labels and title
plt.title('Train and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('v10_loss_graph.png')