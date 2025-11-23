import numpy as np
import torch
from Rubik2x2Env import Rubik2x2Env, apply_move_idx, encode_onehot, is_solved

class MCTSNode:
    def __init__(self, cube, obs, action_mask=None, parent=None, prior=1.0, last_face=None, repeat_count=0):
        self.cube = cube
        self.obs = obs
        self.action_mask = action_mask
        self.parent = parent
        self.prior = float(prior)

        self.children = {}

        self.N = 0
        self.W = 0.0
        self.Q = 0.0

        self.last_face = last_face
        self.repeat_count = repeat_count

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, num_actions=12, c_puct=1.3, num_simulations=200, device="cpu"):
        self.model = model
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.mask_env = Rubik2x2Env(scramble_len=0, max_steps=40, use_action_mask=True)

    def run(self, root_cube, root_obs, root_action_mask=None):
        root = MCTSNode(
            cube=root_cube,
            obs=root_obs,
            action_mask=root_action_mask,
            parent=None,
            prior=1.0,
            last_face=None,
            repeat_count=0,
        )

        for _ in range(self.num_simulations):
            leaf, path = self._select(root)
            value = self._expand_and_evaluate(leaf)
            self._backup(path, value, leaf_node=leaf)

        visit_counts = np.zeros((self.num_actions,), dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.N
        return visit_counts


    def _select(self, node):
        path = []
        current = node
        while not current.is_leaf():
            action, next_node = self._select_child(current)
            path.append((current, action))
            current = next_node
        return current, path

    def _select_child(self, node: MCTSNode):
        total_N = sum(child.N for child in node.children.values())
        best_score = -1e9
        best_action = None
        best_child = None

        for action, child in node.children.items():
            if node.action_mask is not None and not node.action_mask[action]:
                continue

            Q = child.Q
            U = self.c_puct * child.prior * np.sqrt(total_N + 1e-8) / (1 + child.N)
            score = Q + U

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_and_evaluate(self, node: MCTSNode):
        obs_t = torch.from_numpy(node.obs).float().to(self.device)
        with torch.no_grad():
            policy, value = self.model.predict(obs_t)
        policy = policy.cpu().numpy()
        value = float(value.item())

        if is_solved(node.cube):
            return 1.0

        for action in range(self.num_actions):
            if node.action_mask is not None and not node.action_mask[action]:
                continue

            new_cube = apply_move_idx(node.cube, action)
            new_obs = encode_onehot(new_cube)

            curr_face = action // 2
            if node.last_face is not None and node.last_face == curr_face:
                new_repeat = node.repeat_count + 1
            else:
                new_repeat = 1

            self.mask_env.cube = new_cube.copy()
            self.mask_env.steps = 0
            self.mask_env._last_action = action
            self.mask_env._last_face = curr_face
            self.mask_env._last_face_count = new_repeat

            mask = self.mask_env._legal_action_mask()

            child = MCTSNode(
                cube=new_cube,
                obs=new_obs,
                action_mask=mask,
                parent=node,
                prior=policy[action],
                last_face = curr_face,
                repeat_count = new_repeat
            )
            node.children[action] = child

        return value

    def _backup(self, path, value, leaf_node=None):
        for node, action in reversed(path):
            child = node.children[action]
            child.N += 1
            child.W += value
            child.Q = child.W / child.N

        if leaf_node is not None:
            leaf_node.N += 1
            leaf_node.W += value
            leaf_node.Q = leaf_node.W / leaf_node.N