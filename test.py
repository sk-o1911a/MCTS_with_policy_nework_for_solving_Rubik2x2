import torch

from Rubik2x2Env import Rubik2x2Env
from Policy_Value_Net import PolicyValueNet
from MCTS_Core import MCTS
from Action_MTCS import pick_action_from_mcts


def load_model(path: str, device: str):
    model = PolicyValueNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_one_episode(model, device="cpu", scramble_len=15,
                    num_simulations=200, max_steps=50):

    env = Rubik2x2Env(scramble_len=scramble_len, use_action_mask=True)
    obs, info = env.reset()
    cube = env.cube
    print("== start cube ==")
    print(env.as_ascii())


    mcts = MCTS(
        model=model,
        num_actions=env.action_space.n,
        num_simulations=num_simulations,
        device=device,
    )

    for step in range(max_steps):
        action_mask = info.get("action_mask", None)


        visit_counts = mcts.run(cube, obs, action_mask)

        action = pick_action_from_mcts(
            visit_counts,
            mode="greedy",
            temperature=1.0,
        )


        obs, reward, terminated, truncated, info = env.step(action)
        cube = env.cube

        move_name = env.get_action_meanings()[action]
        print(f"\nStep {step+1}: {move_name} ({action})")
        print(env.as_ascii())

        if terminated:
            print("\nSolved!")
            return True

    print("\nNot solved within step limit.")
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "rubik_policy_value.pt"

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, device)


    run_one_episode(
        model,
        device=device,
        scramble_len=5,
        num_simulations=200,
        max_steps=50,
    )


if __name__ == "__main__":
    main()
