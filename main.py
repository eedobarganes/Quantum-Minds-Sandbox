"""
Multi‑Agent Intentional‑Operator Sandbox  v1.1
────────────────────────────────────────────────────────────
Bug‑fix release: removed stray reference to undefined `self` inside
goal‑update routine (NameError).  Behaviour unchanged.

Simulates **N quantum‑conscious agents** that roam a 2‑D arena and
attempt to reach *global coalescence* (maximal mutual fidelity).

Visualisation
─────────────
• Each agent is a circle.  Colour → coherence with the group goal.
• Radius → local fidelity (bigger ≈ more aligned).
• Arrows show velocity.

Controls (while Pygame window is focused)
─────────────────────────────────────────
[SPACE]  Pause / resume
↑ / ↓    Increase / decrease intentional focus f
r        Reset everything (positions + quantum states)
q / ESC  Quit

Dependencies
────────────
Python ≥3.11, numpy, pygame
    pip install numpy pygame

Run Example
───────────
python intentional_operator_multi.py  --agents 8 --dim 32 --focus 0.25
"""

from __future__ import annotations
import argparse, math, random
import numpy as np
import pygame

ℏ = 1.0  # natural units
WIDTH, HEIGHT = 1000, 700
BG = (15, 15, 28)

# ───────────────────── Quantum helpers ─────────────────────────

def random_pure(d: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return vec / np.linalg.norm(vec)

def random_hermitian(d: int, band: int | None = None, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (M + M.conj().T) / 2.0
    if band is not None:
        mask = np.abs(np.subtract.outer(np.arange(d), np.arange(d))) > band
        H[mask] = 0.0
        H = (H + H.conj().T) / 2.0
    return H

def lindblad_step(rho: np.ndarray, H: np.ndarray, Ls: list[np.ndarray], dt: float) -> np.ndarray:
    drho = -1j * (H @ rho - rho @ H)
    for L in Ls:
        drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return rho + dt * drho

def intention_align(rho: np.ndarray, goal_psi: np.ndarray, focus: float) -> np.ndarray:
    """Non‑unitary boost toward |goal><goal| by factor (1+f)."""
    I = np.eye(len(goal_psi), dtype=complex) + focus * np.outer(goal_psi, goal_psi.conj())
    rho2 = I @ rho @ I.conj().T
    return rho2 / np.trace(rho2)

# ───────────────────── Agent class ─────────────────────────────

class Agent:
    COL_PALETTE = (np.array([ 85, 170, 255]),  # low coherence → bluish
                   np.array([240,  90,  40]))  # high coherence → orange

    def __init__(self, idx: int, dim: int, arena: tuple[int,int], rng: np.random.Generator,
                 gamma: float, focus: float):
        self.idx = idx
        self.d = dim
        self.x = rng.uniform(50, arena[0]-50)
        self.y = rng.uniform(50, arena[1]-50)
        angle = rng.uniform(0, 2*math.pi)
        speed = rng.uniform(50, 120)
        self.vx = math.cos(angle)*speed
        self.vy = math.sin(angle)*speed
        self.base_r = 18
        self.focus = focus
        # Quantum parts
        self.H0 = random_hermitian(dim, band=4, seed=idx)
        self.Ls = [math.sqrt(gamma) * np.diag([1 if k==i else -1 for k in range(dim)])
                   for i in range(dim)]
        psi = random_pure(dim, rng)
        self.rho = np.outer(psi, psi.conj())
        self.fid = 0.0  # updated externally

    def update_quantum(self, dt: float, goal_psi: np.ndarray):
        self.rho = lindblad_step(self.rho, self.H0, self.Ls, dt)
        # intention pulse approximated each frame
        self.rho = intention_align(self.rho, goal_psi, self.focus*dt)

    def update_physics(self, agents: list['Agent'], dt: float):
        ax = ay = 0.0
        for other in agents:
            if other is self:
                continue
            dx, dy = other.x - self.x, other.y - self.y
            dist2 = dx*dx + dy*dy + 1e-3
            fid = float(np.real(np.trace(self.rho @ other.rho)))
            ax += fid * dx / dist2
            ay += fid * dy / dist2
        self.vx += ax * dt * 100
        self.vy += ay * dt * 100
        self.vx *= 0.99
        self.vy *= 0.99
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < self.base_r or self.x > WIDTH-self.base_r:
            self.vx *= -1
        if self.y < self.base_r or self.y > HEIGHT-self.base_r:
            self.vy *= -1

    def coherence_colour(self) -> tuple[int,int,int]:
        c = np.clip(self.fid, 0.0, 1.0)
        col = (1-c)*Agent.COL_PALETTE[0] + c*Agent.COL_PALETTE[1]
        return tuple(col.astype(int))

    def draw(self, surf: pygame.Surface):
        pygame.draw.circle(surf, self.coherence_colour(), (int(self.x), int(self.y)),
                           int(self.base_r + 8*self.fid))
        vx, vy = self.vx*0.2, self.vy*0.2
        pygame.draw.line(surf, (200,200,200), (self.x, self.y),
                         (self.x+vx, self.y+vy), 2)

# ───────────────────── Simulation Engine ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi‑Agent Intentional Operator Sandbox")
    parser.add_argument('--agents', type=int, default=6)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--focus', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quantum Consciousness Coalescence Sandbox")
    clock = pygame.time.Clock()

    def spawn_agents():
        return [Agent(i, args.dim, (WIDTH, HEIGHT), rng, args.gamma, args.focus)
                for i in range(args.agents)]
    agents = spawn_agents()

    running = True
    paused = False
    t_last_goal = 0.0
    dt_goal = 0.5  # seconds between goal recomputation
    goal_psi = random_pure(args.dim, rng)

    while running:
        dt = clock.tick(args.fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    for ag in agents:
                        ag.focus = min(1.0, ag.focus + 0.05)
                elif event.key == pygame.K_DOWN:
                    for ag in agents:
                        ag.focus = max(0.0, ag.focus - 0.05)
                elif event.key == pygame.K_r:
                    agents = spawn_agents()
                    goal_psi = random_pure(args.dim, rng)
                    paused = False

        if not paused:
            t_last_goal += dt
            if t_last_goal >= dt_goal:
                fid_matrix = np.zeros((len(agents), len(agents)))
                for i, a in enumerate(agents):
                    for j, b in enumerate(agents):
                        fid_matrix[i, j] = float(np.real(np.trace(a.rho @ b.rho)))
                idx_goal = int(np.argmax(fid_matrix.mean(axis=1)))
                eigvals, eigvecs = np.linalg.eigh(agents[idx_goal].rho)
                goal_psi = eigvecs[:, np.argmax(eigvals)]
                t_last_goal = 0.0
            for ag in agents:
                ag.update_quantum(dt, goal_psi)
            for ag in agents:
                ag.fid = float(np.real(np.trace(ag.rho @ np.outer(goal_psi, goal_psi.conj()))))
            for ag in agents:
                ag.update_physics(agents, dt)

        screen.fill(BG)
        for ag in agents:
            ag.draw(screen)
        font = pygame.font.SysFont("consolas", 16)
        txt = font.render(f"Agents:{len(agents)}  f={agents[0].focus:.2f}  [SPACE]pause  r reset", True, (180,180,180))
        screen.blit(txt, (10, 10))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
