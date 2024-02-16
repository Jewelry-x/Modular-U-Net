import openpyxl
import matplotlib.pyplot as plt
import mplcursors
import os
from config import RESULT_PATH

# Load the workbook
workbook = openpyxl.load_workbook(os.path.join(RESULT_PATH, "progress.xlsx"))
worksheet = workbook.active

# Initialize lists to store data
epochs = []
train_losses = []
val_losses = []
val_ious = []

# Iterate through the rows and extract data
for row in worksheet.iter_rows(min_row=2, values_only=True):  # Assuming data starts from row 2
    epoch, train_loss, val_loss, val_iou, model_saved = row
    epochs.append(int(epoch.split()[1]))  # Extracting the epoch number from "Epoch <number>"
    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))
    val_ious.append(float(val_iou))

# Plotting
plt.figure(figsize=(10, 6))
train_plot, = plt.plot(epochs, train_losses, label='Train Loss')
val_plot, = plt.plot(epochs, val_losses, label='Validation Loss')
iou_plot, = plt.plot(epochs, val_ious, label='Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

# Use mplcursors to display tooltips with data values
cursor = mplcursors.cursor([train_plot, val_plot, iou_plot], hover=True)
@cursor.connect("add")
def on_add(sel):
    x, y = sel.target
    x_rounded = round(x)
    sel.annotation.set_text(f'Epoch: {x_rounded}, Value: {y:.4f}')

plt.show()
