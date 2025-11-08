import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

FACE_NAMES = ["U", "R", "F", "D", "L", "B"]
U, R, F, D, L, B = 0, 1, 2, 3, 4, 5

def solved_cube():
    return np.stack([np.full((2,2), f, dtype=np.int8) for f in range(6)], axis=0)

def rotate_face_clockwise(face):
    return np.rot90(face, -1)

def rotate_face_counter_clockwise(face):
    return np.rot90(face, 1)

def deep_copy_cube(cube):
    return np.copy(cube)

def copy_row(src_face, src_row):
    return src_face[src_row, :].copy()

def copy_col(src_face, src_col):
    return src_face[:, src_col].copy()

####U####
def move_u_cw(cube):
    cube = deep_copy_cube(cube)

    cube[U] = rotate_face_clockwise(cube[U])
    temp_r_row0 = copy_row(cube[R], 0)
    temp_b_row0 = copy_row(cube[B], 0)
    temp_l_row0 = copy_row(cube[L], 0)
    temp_f_row0 = copy_row(cube[F], 0)

    cube[F][0, :] = temp_r_row0
    cube[R][0, :] = temp_b_row0
    cube[B][0, :] = temp_l_row0
    cube[L][0, :] = temp_f_row0
    return cube

####U'####
def move_u_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[U] = rotate_face_counter_clockwise(cube[U])
    temp_l_row0 = copy_row(cube[L], 0)
    temp_b_row0 = copy_row(cube[B], 0)
    temp_r_row0 = copy_row(cube[R], 0)
    temp_f_row0 = copy_row(cube[F], 0)

    cube[F][0, :] = temp_l_row0
    cube[L][0, :] = temp_b_row0
    cube[B][0, :] = temp_r_row0
    cube[R][0, :] = temp_f_row0
    return cube

###R###
def move_r_cw(cube):
    cube = deep_copy_cube(cube)

    cube[R] = rotate_face_clockwise(cube[R])
    temp_u_col1 = copy_col(cube[U], 1)
    temp_f_col1 = copy_col(cube[F], 1)
    temp_d_col1 = copy_col(cube[D], 1)
    temp_b_col0 = copy_col(cube[B], 0)

    cube[U][:, 1] = temp_f_col1
    cube[F][:, 1] = temp_d_col1
    cube[D][:, 1] = temp_b_col0[::-1]
    cube[B][:, 0] = temp_u_col1[::-1]
    return cube

###R'###
def move_r_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[R] = rotate_face_counter_clockwise(cube[R])
    temp_u_col1 = copy_col(cube[U], 1)
    temp_b_col0 = copy_col(cube[B], 0)
    temp_d_col1 = copy_col(cube[D], 1)
    temp_f_col1 = copy_col(cube[F], 1)

    cube[F][:, 1] = temp_u_col1
    cube[U][:, 1] = temp_b_col0[::-1]
    cube[B][:, 0] = temp_d_col1[::-1]
    cube[D][:, 1] = temp_f_col1
    return cube

###F###
def move_f_cw(cube):
    cube = deep_copy_cube(cube)

    cube[F] = rotate_face_clockwise(cube[F])
    temp_l_col1 = copy_col(cube[L], 1)
    temp_r_col0 = copy_col(cube[R], 0)
    temp_d_row0 = copy_row(cube[D], 0)
    temp_u_row1 = copy_row(cube[U], 1)

    cube[U][1, :] = temp_l_col1[::-1]
    cube[D][0, :] = temp_r_col0[::-1]
    cube[L][:, 1] = temp_d_row0
    cube[R][:, 0] = temp_u_row1
    return cube

###F'###
def move_f_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[F] = rotate_face_counter_clockwise(cube[F])
    temp_r_col0 = copy_col(cube[R], 0)
    temp_l_col1 = copy_col(cube[L], 1)
    temp_u_row1 = copy_row(cube[U], 1)
    temp_d_row0 = copy_row(cube[D], 0)

    cube[U][1, :] = temp_r_col0
    cube[D][0, :] = temp_l_col1
    cube[L][:, 1] = temp_u_row1[::-1]
    cube[R][:, 0] = temp_d_row0[::-1]
    return cube

###L###
def move_l_cw(cube):
    cube = deep_copy_cube(cube)

    cube[L] = rotate_face_clockwise(cube[L])
    temp_u_col0 = copy_col(cube[U], 0)
    temp_d_col0 = copy_col(cube[D], 0)
    temp_b_col1 = copy_col(cube[B], 1)
    temp_f_col0 = copy_col(cube[F], 0)

    cube[F][:, 0] = temp_u_col0
    cube[B][:, 1] = temp_d_col0[::-1]
    cube[U][:, 0] = temp_b_col1[::-1]
    cube[D][:, 0] = temp_f_col0
    return cube

###L'###
def move_l_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[L] = rotate_face_counter_clockwise(cube[L])
    temp_d_col0 = copy_col(cube[D], 0)
    temp_u_col0 = copy_col(cube[U], 0)
    temp_f_col0 = copy_col(cube[F], 0)
    temp_b_col1 = copy_col(cube[B], 1)

    cube[F][:, 0] = temp_d_col0
    cube[B][:, 1] = temp_u_col0[::-1]
    cube[U][:, 0] = temp_f_col0
    cube[D][:, 0] = temp_b_col1[::-1]
    return cube

###B###
def move_b_cw(cube):
    cube = deep_copy_cube(cube)

    cube[B] = rotate_face_clockwise(cube[B])
    temp_r_col1 = copy_col(cube[R], 1)
    temp_l_col0 = copy_col(cube[L], 0)
    temp_u_row0 = copy_row(cube[U], 0)
    temp_d_row1 = copy_row(cube[D], 1)

    cube[U][0, :] = temp_r_col1
    cube[D][1, :] = temp_l_col0
    cube[L][:, 0] = temp_u_row0[::-1]
    cube[R][:, 1] = temp_d_row1[::-1]
    return cube

###B'###
def move_b_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[B] = rotate_face_counter_clockwise(cube[B])
    temp_l_col0 = copy_col(cube[L], 0)
    temp_r_col1 = copy_col(cube[R], 1)
    temp_d_row1 = copy_row(cube[D], 1)
    temp_u_row0 = copy_row(cube[U], 0)

    cube[U][0, :] = temp_l_col0[::-1]
    cube[D][1, :] = temp_r_col1[::-1]
    cube[L][:, 0] = temp_d_row1
    cube[R][:, 1] = temp_u_row0
    return cube

###D###
def move_d_cw(cube):
    cube = deep_copy_cube(cube)

    cube[D] = rotate_face_clockwise(cube[D])
    temp_l_row1 = copy_row(cube[L], 1)
    temp_b_row1 = copy_row(cube[B], 1)
    temp_r_row1 = copy_row(cube[R], 1)
    temp_f_row1 = copy_row(cube[F], 1)

    cube[F][1, :] = temp_l_row1
    cube[L][1, :] = temp_b_row1
    cube[B][1, :] = temp_r_row1
    cube[R][1, :] = temp_f_row1

    return cube

####D'####
def move_d_ccw(cube):
    cube = deep_copy_cube(cube)

    cube[D] = rotate_face_counter_clockwise(cube[D])
    temp_r_row1 = copy_row(cube[R], 1)
    temp_b_row1 = copy_row(cube[B], 1)
    temp_l_row1 = copy_row(cube[L], 1)
    temp_f_row1 = copy_row(cube[F], 1)

    cube[F][1, :] = temp_r_row1
    cube[R][1, :] = temp_b_row1
    cube[B][1, :] = temp_l_row1
    cube[L][1, :] = temp_f_row1
    return cube

MOVE_FUNCS = {
    0: ("U",  move_u_cw),
    1: ("U'", move_u_ccw),
    2: ("R",  move_r_cw),
    3: ("R'", move_r_ccw),
    4: ("F",  move_f_cw),
    5: ("F'", move_f_ccw),
    6: ("L",  move_l_cw),
    7: ("L'", move_l_ccw),
    8: ("B",  move_b_cw),
    9: ("B'", move_b_ccw),
    10: ("D",  move_d_cw),
    11: ("D'", move_d_ccw),
}

def apply_move_idx(cube, move_idx):
    return MOVE_FUNCS[move_idx][1](cube)

def scramble(cube, k=8):
    last = None
    applied = 0
    while applied < k:
        move = random.randrange(len(MOVE_FUNCS))
        if last is not None and move // 2 == last // 2 and move != last:
            continue
        cube = apply_move_idx(cube, move)
        last = move
        applied += 1
    return cube

def is_solved(cube):
    for f in range(6):
        if not np.all(cube[f] == cube[f][0,0]):
            return False
    return True

def encode_onehot(cube):
    # (6 faces x 2 x 2) x 6 colors = (24, 6) -> flatten (144,)
    onehot = np.zeros((24, 6), dtype=np.float32)
    flat = cube.reshape(-1)
    for i, c in enumerate(flat):
        onehot[i, int(c)] = 1.0
    return onehot.ravel()

class Rubik2x2Env(gym.Env):
    metadata = {"render_modes": ["ansi", "none"]}
    def __init__(
        self,
        scramble_len: int = 4,
        max_steps: int = 100,
        use_action_mask: bool = True,
        render_mode: str = "none",
        seed: int | None = None,
    ):
        super().__init__()
        self.scramble_len   = int(scramble_len)
        self.max_steps      = int(max_steps)
        self.use_action_mask = bool(use_action_mask)
        self.render_mode    = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(144,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(MOVE_FUNCS))

        # Episode state
        self.cube = solved_cube()
        self.steps = 0
        self._last_action: int | None = None
        self._last_face: int | None = None
        self._last_face_count: int = 0

    # inverse action by use bitwise XOR 1
    @staticmethod
    def _inverse_move_idx(idx: int) -> int:
        return idx ^ 1

    # create mask for legal actions, avoiding repeating the inverse action of the last action
    def _legal_action_mask(self) -> np.ndarray:
        mask = np.ones((self.action_space.n,), dtype=bool)
        if self._last_action is not None:
            inv = self._inverse_move_idx(self._last_action)
            if 0 <= inv < self.action_space.n:
                mask[inv] = False

        if self._last_face is not None and self._last_face_count >= 2:
            f = self._last_face
            face_a = f * 2
            face_b = f * 2 + 1
            if 0 <= face_a < self.action_space.n:
                mask[face_a] = False
            if 0 <= face_b < self.action_space.n:
                mask[face_b] = False

        return mask


    # Create a new scrambled cube and return the initial observation
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.cube = solved_cube()
        self.cube = scramble(self.cube, k=self.scramble_len)
        self.steps = 0
        self._last_action = None
        self._last_face = None
        self._last_face_count = 0

        obs = encode_onehot(self.cube)
        info: dict = {}
        if self.use_action_mask:
            info["action_mask"] = self._legal_action_mask()
        return obs, info


    def step(self, action: int):
        assert self.action_space.contains(action), f"invalid action: {action}"
        # move the cube
        self.cube = apply_move_idx(self.cube, action)
        self.steps += 1

        # each face can be turned at most 3 times in a row
        curr_face = action // 2
        if self._last_face is not None and self._last_face == curr_face:
            self._last_face_count += 1
        else:
            self._last_face = curr_face
            self._last_face_count = 1

        # check if solved
        solved = is_solved(self.cube)
        terminated = solved
        truncated = self.steps >= self.max_steps


        # obs & info
        obs = encode_onehot(self.cube)
        info = {"solved": solved}

        #update last mask
        self._last_action = action
        if self.use_action_mask:
            info["action_mask"] = self._legal_action_mask()

        reward = 0.0
        return obs, reward, terminated, truncated, info

    def as_ascii(self) -> str:
        faces = self.cube

        def row_to_str(row):
            return " ".join(str(int(x)) for x in row)

        lines = []
        # U
        lines.append("     " + row_to_str(faces[U][0]))
        lines.append("     " + row_to_str(faces[U][1]))
        # L F R B
        for r in range(2):
            lines.append(
                row_to_str(faces[L][r]) + "  " +
                row_to_str(faces[F][r]) + "  " +
                row_to_str(faces[R][r]) + "  " +
                row_to_str(faces[B][r])
            )
        # D
        lines.append("     " + row_to_str(faces[D][0]))
        lines.append("     " + row_to_str(faces[D][1]))
        return "\n".join(lines)

    # return the list of action meanings
    def get_action_meanings(self):
        return [MOVE_FUNCS[i][0] for i in range(self.action_space.n)]

    def close(self):
        pass
