# Catheter-U-Net

Customizable Catheter-U-Net model used to optimize the transfer learning between Phantom Data (data on test equipment) and T1-T6 data (videos of catheter procedures). 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

For this project you should have Python 3.9+ installed and preferably CUDA to significantly increase the speed of training.

## Installation

### 1. Clone the repository

Clone the respotitory to your desired location.

### 2.a. Set up virtual environment (Manual, Recommended) 

In order to set up manually the virtual environment you need to run the following commands.
    
1. Run `python -m venv cath` to create the virtual environment.

2. Nagivate into `cath/Scipts` and run the `activate` script.

3. Go to the [Pytorch website](https://pytorch.org/get-started/locally/) and find the suitable Pytorch version and install it by running the command provided.

4. Run `pip install openpyxl opencv-python` to have all required packages.

### 2.b. Set up virtual environment (Automatic)

Run script `setupVENV.sh` found in the `utils/` directory to set up the virtual environment for the project.

## Usage

### 1. Activate virtual environment

Nagivate into the scripts folder in the virtual environment and use `activate` if not already active.

### 2. Set config.py

Check the config.py file to change any settings required such as data location and result location.

### 3. Run the model

Use command `python model.py` to run the model.