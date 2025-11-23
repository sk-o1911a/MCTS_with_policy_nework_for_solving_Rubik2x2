import os
import json
import torch
import matplotlib.pyplot as plt
from Policy_Value_Net import PolicyValueNet
from Self_Play import generate_self_play_data

class ScrambleEvaluator:
    def __init__(
        self,
        checkpoint_path="rubik_policy_value.pt",
        result_json_path=os.path.join("training_logs", "eval_scramble_result.json"),
        n_runs=100,
        device=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.result_json_path = result_json_path
        self.n_runs = n_runs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.results = []

    def load_model(self):
        model = PolicyValueNet().to(self.device)
        if os.path.exists(self.checkpoint_path):
            print(f"[evaluate] Loading checkpoint: {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        self.model = model

    def evaluate_scramble_range(self, min_len=1, max_len=20, num_simulations=1000, max_episode_steps=40):
        if self.model is None:
            self.load_model()
        self.results = []
        for scramble_len in range(min_len, max_len + 1):
            print(f"==> Evaluating scramble_len={scramble_len} ...")
            _, solve_rate = generate_self_play_data(
                model=self.model,
                num_episodes=self.n_runs,
                max_episode_steps=max_episode_steps,
                num_simulations=num_simulations,
                scramble_len=scramble_len,
                select_mode="greedy",
                temperature=0.3,
                device=self.device,
            )
            self.results.append({
                "scramble_len": scramble_len,
                "solve_rate": solve_rate
            })
        os.makedirs(os.path.dirname(self.result_json_path), exist_ok=True)
        with open(self.result_json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[evaluate] Saved results to {self.result_json_path}")

    def plot_scramble_results(self, json_path=None):
        path = json_path or self.result_json_path
        if not self.results and os.path.exists(path):
            with open(path, "r") as f:
                self.results = json.load(f)
        if not self.results:
            print("[evaluate] No results available to plot.")
            return
        scramble_lengths = [entry["scramble_len"] for entry in self.results]
        solve_rates = [entry["solve_rate"] * 100 for entry in self.results]  # convert to %

        plt.figure(figsize=(10, 6))
        plt.bar(scramble_lengths, solve_rates, color='skyblue', width=0.8)
        plt.xlabel("Scramble Length", fontsize=13)
        plt.ylabel("Solve Rate (%)", fontsize=13)
        plt.title("Solve Rate (%) by Scramble Length", fontsize=16)
        plt.xticks(scramble_lengths)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("training_logs/eval_bar_chart.png", dpi=150)
        plt.show()
        print("[evaluate] Saved bar chart to training_logs/eval_bar_chart.png")

if __name__ == "__main__":
    evaluator = ScrambleEvaluator()
    evaluator.evaluate_scramble_range()
    evaluator.plot_scramble_results()