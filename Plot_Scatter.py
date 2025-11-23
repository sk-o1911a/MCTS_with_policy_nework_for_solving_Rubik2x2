import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict
import numpy as np


class MetricsLogger:
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.iterations: List[int] = []
        self.losses: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.solve_rates: List[float] = []
        self.scramble_lengths: List[int] = []
        self.num_samples: List[int] = []

    def log_iteration(
            self,
            iteration: int,
            loss: float,
            policy_loss: float,
            value_loss: float,
            solve_rate: float,
            scramble_len: int,
            num_samples: int
    ):
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.solve_rates.append(solve_rate)
        self.scramble_lengths.append(scramble_len)
        self.num_samples.append(num_samples)

    def save_json(self, filename: str = "metrics.json"):
        data = {
            "iterations": self.iterations,
            "losses": self.losses,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "solve_rates": self.solve_rates,
            "scramble_lengths": self.scramble_lengths,
            "num_samples": self.num_samples
        }
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[logger] saved metrics to {filepath}")

    def load_json(self, filename: str = "metrics.json"):
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.iterations = data["iterations"]
            self.losses = data["losses"]
            self.policy_losses = data["policy_losses"]
            self.value_losses = data["value_losses"]
            self.solve_rates = data["solve_rates"]
            self.scramble_lengths = data["scramble_lengths"]
            self.num_samples = data["num_samples"]
            print(f"[logger] loaded metrics from {filepath}")
            return True
        return False

    def plot_all(self, filename: str = "training_metrics.png", show: bool = False):

        if len(self.iterations) == 0:
            print("[logger] no data to plot")
            return

        fig = plt.figure(figsize=(16, 10))

        # Total Loss plot
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.iterations, self.losses, linewidth=2, color='blue', marker='o', markersize=3)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('Total Loss', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Solve rate plot
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.iterations, [sr * 100 for sr in self.solve_rates],
                 linewidth=2, marker='o', markersize=3, color='green')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Solve Rate (%)', fontsize=12)
        ax2.set_title('Solve Rate', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])

        # Policy Loss plot
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(self.iterations, self.policy_losses,
                 linewidth=2, marker='s', markersize=3, color='orange')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Policy Loss', fontsize=12)
        ax3.set_title('Policy Loss', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Value Loss plot
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(self.iterations, self.value_losses,
                 linewidth=2, marker='^', markersize=3, color='red')
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Value Loss', fontsize=12)
        ax4.set_title('Value Loss', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)

        filepath = os.path.join(self.log_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')


        if show:
            plt.show()
        plt.close()
