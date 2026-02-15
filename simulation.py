import numpy as np
import random
import time
import os
import json
from scipy.spatial import distance

WIDTH, HEIGHT = 20.0, 20.0
DT = 0.1
NUM_AGENTS_BASE = 50
MAX_SPEED = 3.5

TAU = 0.5
A_SOC, B_SOC = 2.0, 0.08
K_AG = 1.2e5

class Agent:
    def __init__(self, x, y, goal_x, goal_y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.zeros(2)
        self.prev_vel = np.zeros(2)
        self.goal = np.array([goal_x, goal_y], dtype=float)
        self.radius = 0.25
        self.mass = 80.0
        self.desired_speed = random.uniform(1.0, 1.6)

    def desired_force(self):
        direction = self.goal - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        target_vel = direction * self.desired_speed
        return (self.mass * (target_vel - self.vel)) / TAU

    def social_force(self, others):
        force = np.zeros(2)
        for other in others:
            if other is self:
                continue
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist == 0:
                continue
            overlap = self.radius + other.radius - dist
            n = diff / dist
            f_soc = A_SOC * np.exp(overlap / B_SOC) * n
            if overlap > 0:
                f_soc += K_AG * overlap * n
            force += f_soc
        return force

    def update(self, force):
        self.prev_vel = self.vel.copy()
        acc = force / self.mass
        self.vel += acc * DT
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED
        self.pos += self.vel * DT

class Simulation:
    def __init__(self):
        self.agents = []

    def reset_scenario(self, scenario_type):
        self.agents = []

        if scenario_type == "safe":
            num_agents = random.randint(40, 70)

        else:
            num_agents = random.randint(70, 120)

        for _ in range(num_agents):
            x = random.uniform(1, WIDTH - 1)
            y = random.uniform(1, HEIGHT - 1)
            goal_x = random.choice([0, WIDTH])
            goal_y = random.choice([0, HEIGHT])
            agent = Agent(x, y, goal_x, goal_y)

            if scenario_type == "safe":
                agent.desired_speed = random.uniform(0.8, 1.8)

            else:
                agent.desired_speed = random.uniform(1.5, 3.0)

            self.agents.append(agent)

    def calculate_metrics(self):

        area = WIDTH * HEIGHT
        density = len(self.agents) / area

        speeds = np.array([np.linalg.norm(a.vel) for a in self.agents])
        mean_speed = np.mean(speeds)
        speed_var = np.var(speeds)
        flux = np.sum(speeds)

        accels = np.array([
            np.linalg.norm(a.vel - a.prev_vel) / DT
            for a in self.agents
        ])
        acceleration = np.mean(accels)

        positions = np.array([a.pos for a in self.agents])
        centroid = np.mean(positions, axis=0)
        radial_spread = np.mean(np.linalg.norm(positions - centroid, axis=1))

        if len(self.agents) > 1:
            dists = distance.pdist(positions)
            density_gradient = np.std(dists)
        else:
            density_gradient = 0.0

        instability_score = (
            0.3 * (speed_var / 1.5) +
            0.3 * (acceleration / 4.0) +
            0.2 * (density / 3.0) +
            0.2 * (flux / 150.0)
        )

        instability_score += random.uniform(-0.15, 0.15)

        is_shockwave = bool(instability_score > 0.75)

        return {
            "density": density,
            "mean_speed": mean_speed,
            "speed_variance": speed_var,
            "radial_spread": radial_spread,
            "density_gradient": density_gradient,
            "acceleration": acceleration,
            "flux": flux,
            "is_shockwave": is_shockwave
        }

    def step(self):
        forces = []
        for agent in self.agents:
            f = agent.desired_force()
            f += agent.social_force(self.agents)
            forces.append(f)

        for i, agent in enumerate(self.agents):
            agent.update(forces[i])

    def run_scenario(self, steps, scenario_type):

        scenario_data = []
        start_time = time.time()

        for frame in range(steps):

            if scenario_type == "panic" and frame > steps // 3:
                for agent in self.agents:
                    agent.desired_speed += random.uniform(0.0, 0.02)

            self.step()
            metrics = self.calculate_metrics()

            scenario_data.append({
                "ts": start_time + frame * DT,
                "frame": frame,
                "label": "SHOCKWAVE" if metrics["is_shockwave"] else "SAFE",
                **metrics
            })

        return scenario_data

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="synthetic_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sim = Simulation()

    print("Generating improved synthetic dataset...")

    for i in range(args.scenarios):

        scenario_type = random.choice(["safe", "panic"])
        sim.reset_scenario(scenario_type)
        data = sim.run_scenario(args.steps, scenario_type)

        with open(os.path.join(args.output_dir, f"scenario_{i:04d}.json"), "w") as f:
            json.dump(data, f)

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{args.scenarios} scenarios generated")

    print("Done.")
