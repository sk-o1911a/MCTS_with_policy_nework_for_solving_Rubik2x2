import pygame
import sys
import torch

from Rubik2x2Env import (
    Rubik2x2Env,
    solved_cube,
    MOVE_FUNCS,
    encode_onehot,
    is_solved,
)
from Policy_Value_Net import PolicyValueNet
from MCTS_Core import MCTS
from Action_MCTS import pick_action_from_mcts

pygame.init()

WIDTH, HEIGHT = 900, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rubik 2x2 Controller")
FONT = pygame.font.SysFont("consolas", 18)
CLOCK = pygame.time.Clock()

COLORS = {
    0: (255, 255, 255),  # U
    1: (255, 165, 0),    # R
    2: (0, 255, 0),      # F
    3: (255, 255, 0),    # D
    4: (0, 0, 255),      # L
    5: (255, 0, 0),      # B
}

TILE = 40
MARGIN = 4

FACE_POS = {
    "U": (WIDTH // 2 - 1 * TILE, 40),
    "L": (WIDTH // 2 - 3 * TILE - 2 * TILE, 40 + 2 * TILE + 20),
    "F": (WIDTH // 2 - 1 * TILE,            40 + 2 * TILE + 20),
    "R": (WIDTH // 2 + 1 * TILE + 20,       40 + 2 * TILE + 20),
    "B": (WIDTH // 2 + 3 * TILE + 40,       40 + 2 * TILE + 20),
    "D": (WIDTH // 2 - 1 * TILE,            40 + 4 * TILE + 40),
}


# ========================= UI helpers =========================
class Button:
    def __init__(self, x, y, w, h, text, cb):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.cb = cb

    def draw(self, surf):
        pygame.draw.rect(surf, (220, 220, 220), self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 1)
        txt = FONT.render(self.text, True, (0, 0, 0))
        surf.blit(txt, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.cb:
                    self.cb()


class InputBox:
    def __init__(self, x, y, w, h, text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isdigit():
                    self.text += event.unicode

    def draw(self, surf):
        pygame.draw.rect(surf, (255, 255, 255), self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)
        txt = FONT.render(self.text, True, (0, 0, 0))
        surf.blit(txt, (self.rect.x + 5, self.rect.y + 5))

    def get_value(self, default=4):
        try:
            v = int(self.text)
            return max(1, v)
        except:
            return default


# ========================= draw cube =========================
def draw_face(surface, cube, face_idx, face_name):
    ox, oy = FACE_POS[face_name]
    face = cube[face_idx]
    for r in range(2):
        for c in range(2):
            color_id = int(face[r, c])
            color = COLORS.get(color_id, (128, 128, 128))
            x = ox + c * (TILE + MARGIN)
            y = oy + r * (TILE + MARGIN)
            pygame.draw.rect(surface, color, (x, y, TILE, TILE))
            pygame.draw.rect(surface, (0, 0, 0), (x, y, TILE, TILE), 2)
    label = FONT.render(face_name, True, (0, 0, 0))
    surface.blit(label, (ox, oy - 18))


def draw_cube(surface, env):
    cube = env.cube
    draw_face(surface, cube, 0, "U")
    draw_face(surface, cube, 4, "L")
    draw_face(surface, cube, 2, "F")
    draw_face(surface, cube, 1, "R")
    draw_face(surface, cube, 5, "B")
    draw_face(surface, cube, 3, "D")


def solve_with_mcts(env, model, device="cpu", num_simulations=200, max_steps=40):
    temp_env = Rubik2x2Env(scramble_len=1, max_steps=40, use_action_mask=True)
    temp_env.cube = env.cube.copy()
    temp_env.steps = 0
    temp_env._last_action = None

    obs = encode_onehot(temp_env.cube)
    action_mask = temp_env._legal_action_mask()

    mcts = MCTS(
        model=model,
        num_actions=temp_env.action_space.n,
        num_simulations=num_simulations,
        device=device,
    )

    actions = []
    names = []

    for _ in range(max_steps):
        if is_solved(temp_env.cube):
            break

        visit_counts = mcts.run(temp_env.cube, obs, action_mask)
        act = pick_action_from_mcts(visit_counts, mode="greedy", temperature=1.0)

        obs, _, terminated, truncated, info = temp_env.step(act)
        action_mask = info.get("action_mask", None)

        actions.append(act)
        names.append(MOVE_FUNCS[act][0])

        if terminated:
            break

    formula = " ".join(names)
    return actions, formula


# ========================= main =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PolicyValueNet().to(device)
    try:
        state = torch.load("rubik_policy_value.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("Loaded rubik_policy_value.pt")
    except Exception as e:
        print("Không load được rubik_policy_value.pt:", e)

    env = Rubik2x2Env(scramble_len=4, max_steps=40, use_action_mask=True)

    input_scramble = InputBox(30, 300, 80, 30, text="4")

    buttons = []

    move_labels = [
        ("U", 0), ("U'", 1),
        ("R", 2), ("R'", 3),
        ("F", 4), ("F'", 5),
        ("L", 6), ("L'", 7),
        ("B", 8), ("B'", 9),
        ("D", 10), ("D'", 11),
    ]

    bx, by = 30, 50
    for (label, act) in move_labels:
        def make_cb(a=act):
            def cb():
                env.step(a)
            return cb
        buttons.append(Button(bx, by, 60, 26, label, make_cb()))
        by += 32
        if by > 50 + 32 * 6:
            by = 50
            bx += 70

    def do_scramble():
        nonlocal solve_formula, solve_moves, play_solve
        k = input_scramble.get_value(default=4)
        env.scramble_len = k
        env.reset()
        solve_formula = ""
        solve_moves = []
        play_solve = False
    buttons.append(Button(30, 350, 80, 30, "Scram", do_scramble))

    def do_reset():
        nonlocal solve_formula, solve_moves, play_solve
        env.cube = solved_cube()
        env.steps = 0
        solve_formula = ""
        solve_moves = []
        play_solve = False
    buttons.append(Button(120, 350, 80, 30, "Reset", do_reset))

    solve_moves = []
    solve_formula = ""
    play_solve = False
    next_move_time = 0

    def do_solve():
        nonlocal solve_moves, solve_formula, play_solve, next_move_time
        actions, formula = solve_with_mcts(
            env,
            model,
            device=device,
            num_simulations=1000,
            max_steps=40,
        )
        solve_moves = actions
        solve_formula = formula
        play_solve = True
        next_move_time = pygame.time.get_ticks() + 150  # 1s sau bắt đầu xoay
    buttons.append(Button(210, 350, 80, 30, "Solve", do_solve))

    running = True
    while running:
        CLOCK.tick(30)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            input_scramble.handle_event(event)
            for b in buttons:
                b.handle_event(event)

        if play_solve and len(solve_moves) > 0 and now >= next_move_time:
            act = solve_moves.pop(0)
            env.step(act)
            next_move_time = now + 1000
        if play_solve and len(solve_moves) == 0:
            play_solve = False

        SCREEN.fill((210, 210, 210))

        for b in buttons:
            b.draw(SCREEN)

        input_scramble.draw(SCREEN)
        lbl = FONT.render("Scram steps:", True, (0, 0, 0))
        SCREEN.blit(lbl, (30, 275))

        draw_cube(SCREEN, env)

        formula_label = FONT.render("Formula: " + solve_formula, True, (0, 0, 0))
        SCREEN.blit(formula_label, (30, 400))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
