DATA = "Phantom"
TRAIN = True
LOAD = True
TEST = True

# Index of datasets to test on
TEST_ON = [0, 1]

# If true removes pooling layers from the first convolutional layer, much slower
REVERSE_POOL = True
POOL = 4


LEARNING_RATE = 1e-2
TOTAL_EPOCHS = 500
IMAGE_SIZE = 256

# Stop model early if there is no lower IoU than currently saved
EARLY_STOPPING = True
EARLY_STOPPING_COUNT = 50

CREATE_TEST_MASK = True

# Create automatically directory structure for current model being training
CREATE_FOLDER = True

# Plays notification when training is over for Windows systems
NOTIFY = True

### DATA LOADING

# General data directory
DATA_PATH = "data"

# Name of each dataset
TRAINING_DATA = ["Phantom", "T1T6"]
TESTING_DATA = ["Phantom", "T1T6"]

# Location of each respective training and testing directory for the datasets
TRAINING_DATA_LOCATION = ["PTrain", "TTrain"]
TESTING_DATA_LOCATION = ["PTest", "TTest"]

# Location of training and testing masks for datasets
TRAINING_DATA_MASK_LOCATION = ["PTrain_label", "TTrain_label"]
TESTING_DATA_MASK_LOCATION = ["PTest_label", "TTest_label"]

# File format for image and mask
IMAGE_DEFINITION = "frame_%04d.npy"
MASK_DEFINITION = "frame_%04d.npy"

# Number of images and masks
TRAINING_DATA_COUNT = [1400, 845]
TESTING_DATA_COUNT = [600, 362]

# Result location
RESULT_PATH = "./result"

# Transformations
TRANSFORM = False

HORIZONTAL_FLIP = False
VERTICAL_FLIP = False
SHEAR = False
GAUSSIAN_BLUR = False
