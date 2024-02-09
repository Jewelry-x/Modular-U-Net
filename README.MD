# Catheter-U-Net

Customizable Catheter-U-Net model used to optimized the transfer learning between Phantom Data (generated) and T1-T6 data (real world data). 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

For this project you should have Python 3.9+ installed and preferably CUDA to significantly increase the speed of training.

## Installation

1. Clone the repository:

Clone the respotitory to your desired location

2. Set up virtual environment

Once installed, run script 'setupVENV.sh' to set up the virtual environment for the project.

## Usage

1. Activate virtual environment

Nagivate into the scripts folder in the virtual environment and use `activate`.

2. Check config.py

Check the config.py file to change any settings required such as data location and result location.

3. Run the model

Use command 'python model.py' to run the model.