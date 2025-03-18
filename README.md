# DA6401: Introduction to Deep Learning - Assignment 1

## Assignment Overview
This repository contains the implementation of **Assignment 1** for the course **DA6401: Introduction to Deep Learning** by **DA24S006 (Ashutosh Patidar)**.

### 📄 Assignment Report
The detailed assignment report is available at:  
[DA6401 Assignment Report](https://wandb.ai/da24s006-indian-institue-of-technology-madras-/Question_4/reports/DA6401-Assignment-report-by-DA24S006--VmlldzoxMTgyOTU5NQ)

### 📂 Repository Link  
[Assignment Repository](https://github.com/asu2304/DA6401-Introduction-to-Deep-Learning-Assignment_1)

---

## 🔧 Installation
Ensure that all dependencies are installed before running the scripts. To install the required dependencies, execute:  
```sh
pip install -r requirements.txt
```

### Required Dependencies
Please install the following specific versions of the dependencies before running the code:
```sh
python==3.11.11
wandb==0.19.6
tensorflow==2.18.0
keras==3.8.0
numpy==1.26.4
matplotlib==3.9.3
```
You can add these to your `requirements.txt` file or install them individually using:
```sh
pip install python==3.11.11 wandb==0.19.6 tensorflow==2.18.0 keras==3.8.0 numpy==1.26.4 matplotlib==3.9.3
```

---

## 🚀 Running the Code

### 🔹 Training with `train.py`
The `train.py` script supports various command-line arguments to customize the training process. Run it using Python with the following format:
```sh
python train.py [arguments]
```

### Supported Arguments
| Argument | Short Form | Default Value | Description |
|----------|-----------|--------------|-------------|
| `--wandb_project` | `-wp` | myprojectname | Project name for Weights & Biases tracking |
| `--wandb_entity` | `-we` | myname | Weights & Biases entity |
| `--dataset` | `-d` | fashion_mnist | Dataset selection (`mnist` or `fashion_mnist`) |
| `--epochs` | `-e` | 1 | Number of training epochs |
| `--batch_size` | `-b` | 4 | Batch size for training |
| `--loss` | `-l` | cross_entropy | Loss function (`mean_squared_error` or `cross_entropy`) |
| `--optimizer` | `-o` | sgd | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `--learning_rate` | `-lr` | 0.1 | Learning rate |
| `--momentum` | `-m` | 0.5 | Momentum for `momentum` and `nag` optimizers |
| `--beta` | `-beta` | 0.5 | Beta for `rmsprop` optimizer |
| `--beta1` | `-beta1` | 0.5 | Beta1 for `adam` and `nadam` optimizers |
| `--beta2` | `-beta2` | 0.5 | Beta2 for `adam` and `nadam` optimizers |
| `--epsilon` | `-eps` | 0.000001 | Epsilon for optimizers |
| `--weight_decay` | `-w_d` | 0.0 | Weight decay for optimizers |
| `--weight_init` | `-w_i` | random | Weight initialization (`random` or `Xavier`) |
| `--num_layers` | `-nhl` | 1 | Number of hidden layers |
| `--hidden_size` | `-sz` | 4 | Number of hidden neurons per layer |
| `--activation` | `-a` | sigmoid | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |

### Example Usage:
```sh
python train.py --wandb_project "my_deep_learning_project" --wandb_entity "ashutosh_patidar" --dataset mnist --epochs 5 --batch_size 32 --optimizer adam --learning_rate 0.01
```

---

### 🔹 Running the Hyperparameter Sweep with `question4(sweep_run).py`
To perform a hyperparameter sweep, use the following command:
```sh
python question4(sweep_run).py
```
This script is designed to execute a sweep run for hyperparameter tuning, leveraging Weights & Biases for tracking.

---

## 📁 Repository Contents
This repository contains the following question-specific Python files:
- `question1.py`  
- `question2.py`  
- `question4.py`  
- `question7.py`  

---

## 📌 Notes
- Ensure your environment is set up with the specified Python version and dependencies before running any scripts.
- Refer to the [Assignment Report](https://wandb.ai/da24s006-indian-institue-of-technology-madras-/Question_4/reports/DA6401-Assignment-report-by-DA24S006--VmlldzoxMTgyOTU5NQ) for detailed explanations and results.
