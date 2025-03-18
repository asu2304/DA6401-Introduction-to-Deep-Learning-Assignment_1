Assignment 1 - Deep Learning (DA6401: Introduction to Deep Learning)

Overview

This repository contains the implementation of Assignment 1 for the course DA6401: Introduction to Deep Learning at IIT Madras.

Assignment Report: View Report

GitHub Repository: Access Repository

Installation Instructions

To set up the required environment, install all dependencies from requirements.txt using the following command:

pip install -r requirements.txt

Dependencies

The project requires the following dependencies:

TensorFlow 2.19.0

PyTorch 2.6.0

NumPy 2.1.3

Pandas 2.2.3

Matplotlib 3.10.1

scikit-learn

Weights & Biases (wandb)

For the complete list of dependencies, refer to requirements.txt.

Running the Code

Running the Sweep

To execute the sweep configuration, run the following command:

python question4(sweep_run).py

Training the Model

To execute the training script, run:

python train.py

Code Structure

The repository contains separate script files implementing solutions for different questions. Each file is named accordingly to match the question number. All hyperparameter configurations are implemented in train.py, and the execution steps are outlined in the assignment report.

For further details, refer to the assignment report linked above.

