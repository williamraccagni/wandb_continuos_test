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
