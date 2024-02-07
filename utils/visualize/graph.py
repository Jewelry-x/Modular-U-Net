import matplotlib.pyplot as plt

# Replace 'your_loss_file.txt' with the actual path to your loss file
loss_file_path = 'loss.txt'

# Read the loss values from the text file
with open(loss_file_path, 'r') as file:
    lines = file.readlines()

# Extract loss values and step numbers
steps = []
loss_values = []

for line in lines:
    # Skip lines that do not contain numerical values
    if 'loss' not in line:
        continue

    # Split the line into a list of values
    values = line.strip().split()

    # Extract step and loss from the list
    if len(values) >= 3:
        step, loss = values[1], values[3]
        steps.append(int(step))
        loss_values.append(float(loss))

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(steps, loss_values, label='Loss', color='blue')
plt.title('Training Loss Over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
