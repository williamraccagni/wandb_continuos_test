# Dummy Model Training with W&B Resume Support

This script demonstrates how to integrate Weights & Biases (wandb) into a simulated training loop, with built-in support for automatic resume in case of interruptions. It uses a dummy model (DummyModel) to simulate training and validation loss.

## Key Features

- Automatic logging of training and validation metrics with wandb
- Training resume support using a resume.json file
- Saves current state (run_id, epoch, completion flag)
- Supports multi-epoch training

## 📁 Project Structure

```graphql
.
├── dummy_model.py        # Contains the DummyModel class with train_step and validate methods
├── resume.json           # Resume state file (auto-generated)
├── train.py              # Main script (this file)
```

## 📦 Requirements
- Python 3.7+
- [wandb](https://docs.wandb.ai/quickstart)

Install dependencies with:

```bash
pip install wandb
```

## DummyModel
Make sure you have a file named dummy_model.py in the same directory with the following structure (example):

```python
import random

class DummyModel:
    def __init__(self):
        self.state = 0.0

    def train_step(self):
        loss = 1 / (self.state + 1) + random.uniform(-0.1, 0.1)
        self.state += 1
        return loss

    def validate(self):
        val_loss = 1 / (self.state + 2) + random.uniform(-0.1, 0.1)
        return val_loss

```

## ⚙️ Configuration
In the `train.py` file, you can customize the following:

```python
PROJECT = "wandb_continuos_test"              # Your W&B project name
ENTITY = "YOUR ENTITY"                        # Your W&B username or team
RUN_NAME = "dummy-run"                        # Base run name
```

## ▶️ Running the Script
To start training:

```bash
python train.py
```

If a previous run was interrupted, it will automatically resume from where it left off using the information stored in `resume.json`. If the file does not exist, a new run will start.

## 📌 Notes

- The `resume.json` file is updated after each completed epoch.
- Once training finishes, the `"completed"` flag is set to `true`, and a new run ID will be generated on the next launch.

## Resetting
To force a fresh run from epoch 0, just delete the `resume.json` file:

```bash
rm resume.json
```

## 📈 W&B Output
Each epoch logs:
- `train/loss` for each training step
- `val/loss` once per epoch

All runs will be visible in your wandb workspace.