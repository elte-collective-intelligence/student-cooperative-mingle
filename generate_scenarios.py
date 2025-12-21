"""
Generate GIFs for multiple room/agent scenarios.
"""

import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from tensordict import TensorDict
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import CollisionAvoidanceReward, StayInRoomReward, GetToRoomReward

DEVICE = torch.device('cpu')


def get_reward_modules():
    modules = [
        CollisionAvoidanceReward(min_distance=0.5, penalty=1.0, phase_mode='claiming'),
        GetToRoomReward(max_reward=15.0, phase_mode='claiming'),
        StayInRoomReward(max_reward=20.0, outside_penalty=3.0, overfill_penalty=5.0, phase_mode='claiming'),
    ]
    for m in modules:
        m._activate()
    return modules


class ObservationNormalizer:
    def __init__(self, obs_dim, device):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 0
        self.device = device

    def update(self, obs):
        if obs.numel() == 0:
            return
        obs_flat = obs.reshape(-1, obs.shape[-1])
        batch_count = obs_flat.shape[0]
        if batch_count < 2:
            return
        batch_mean = obs_flat.mean(dim=0)
        batch_var = obs_flat.var(dim=0, unbiased=False) + 1e-8
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            self.var = (self.var * self.count + batch_var * batch_count +
                       delta**2 * self.count * batch_count / total_count) / total_count
            self.count = total_count

    def normalize(self, obs):
        if self.count == 0:
            return obs
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


class ScenarioPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, n_agents=6):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.agent_embed_dim = 8
        self.agent_embedding = nn.Embedding(n_agents, self.agent_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + self.agent_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.3)

    def forward(self, obs):
        was_flat = False
        if obs.dim() == 2:
            if obs.shape[0] == self.n_agents:
                obs = obs.unsqueeze(0)
            elif obs.shape[0] % self.n_agents == 0:
                was_flat = True
                batch_size = obs.shape[0] // self.n_agents
                obs = obs.view(batch_size, self.n_agents, -1)
            else:
                was_flat = True
                n_samples = obs.shape[0]
                agent_ids = torch.arange(n_samples, device=obs.device) % self.n_agents
                agent_embeds = self.agent_embedding(agent_ids)
                obs_with_id = torch.cat([obs, agent_embeds], dim=-1)
                hidden = self.net(obs_with_id)
                adjustment = torch.tanh(self.action_head(hidden))
                room_dir = obs[:, 5:7]
                loc = room_dir * 0.25 + adjustment * 0.05
                scale = self.log_std.exp().expand_as(loc)
                return loc, scale

        batch_size, n_agents, obs_dim = obs.shape
        agent_ids = torch.arange(n_agents, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        agent_embeds = self.agent_embedding(agent_ids)
        obs_with_id = torch.cat([obs, agent_embeds], dim=-1)
        room_dir = obs[:, :, 5:7]
        obs_flat = obs_with_id.view(-1, obs_dim + self.agent_embed_dim)
        hidden = self.net(obs_flat)
        adjustment = torch.tanh(self.action_head(hidden))
        adjustment = adjustment.view(batch_size, n_agents, -1)
        loc = room_dir * 0.25 + adjustment * 0.05
        scale = self.log_std.exp().expand_as(loc)
        if was_flat:
            return loc.view(-1, self.action_dim), scale.view(-1, self.action_dim)
        return loc.squeeze(0), scale.squeeze(0)


def generate_scenario_gif(n_agents, n_rooms, room_capacity, output_path, use_communication=False):
    print(f'Generating: {n_agents} agents, {n_rooms} rooms, capacity {room_capacity}')

    env = MingleEnv(
        n_agents=n_agents,
        n_rooms=n_rooms,
        room_capacity=room_capacity,
        max_steps=300,
        phase_mode='claiming',
        reward_modules=get_reward_modules(),
    )
    env.enable_communication = use_communication

    obs_dim = 18 if use_communication else 14
    policy = ScenarioPolicy(obs_dim, n_agents=n_agents).to(DEVICE)
    obs_normalizer = ObservationNormalizer(obs_dim, DEVICE)

    gif_frames = []
    td = env.reset()
    obs_raw = td['observation'].to(DEVICE)

    config_text = f'{n_agents} Agents, {n_rooms} Rooms\nCapacity: {room_capacity}/room'
    if use_communication:
        config_text += '\n+ Communication'

    for step in range(150):
        obs_normalizer.update(obs_raw.unsqueeze(0))
        obs = obs_normalizer.normalize(obs_raw)

        with torch.no_grad():
            loc, scale = policy(obs.unsqueeze(0))
            noise = torch.randn_like(loc) * scale * 0.5
            actions = (loc + noise).clamp(-0.3, 0.3)

        step_td = TensorDict({'action': actions.cpu()}, batch_size=[])
        next_td = env._step(step_td)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.set_title(f'Scenario: {n_agents}A/{n_rooms}R | Step {step}', fontsize=14)

        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color='gray', linestyle='--'))
        ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color='lightsalmon', alpha=0.3))

        for i, (room_pos, occ) in enumerate(zip(env.room_positions.cpu().numpy(), env.room_occupancy.cpu().numpy())):
            color = 'green' if occ >= room_capacity else 'yellow' if occ > 0 else 'lightgreen'
            ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=True, color=color, alpha=0.3))
            ax.add_artist(plt.Circle(room_pos, env.room_radius, fill=False, color='green'))
            ax.text(room_pos[0], room_pos[1], f'R{i}: {int(occ)}/{room_capacity}',
                    ha='center', va='center', fontsize=10, fontweight='bold')

        positions = env.agent_positions.cpu().numpy()
        colors_list = plt.cm.tab10(np.linspace(0, 1, n_agents))

        for i, pos in enumerate(positions):
            if use_communication and hasattr(env, 'is_leader') and env.is_leader[i]:
                marker = 's'
                label = f'L{i}'
            else:
                marker = 'o'
                label = f'A{i}'
            ax.scatter(pos[0], pos[1], c=[colors_list[i]], s=150, marker=marker,
                      edgecolors='black', linewidths=2, zorder=5)
            ax.annotate(label, (pos[0], pos[1] + 0.7), ha='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.text(0.98, 0.98, config_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='gray'))

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        gif_frames.append(frame[..., :3])
        plt.close(fig)

        if next_td['done'].any() or next_td['terminated'].any():
            break
        obs_raw = next_td['observation'].to(DEVICE)

    imageio.mimsave(str(output_path), gif_frames, fps=10, loop=0)
    print(f'  Saved: {output_path}')


def main():
    # Create output directory
    output_dir = Path('contribution_tests_and_comparisions/scenarios')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate different scenarios
    scenarios = [
        # (n_agents, n_rooms, room_capacity, use_communication)
        (4, 2, 2, False),   # Small: 4 agents, 2 rooms
        (6, 3, 2, False),   # Medium: 6 agents, 3 rooms (standard)
        (8, 4, 2, False),   # Large: 8 agents, 4 rooms
        (6, 2, 3, False),   # Dense: 6 agents, 2 rooms, capacity 3
        (6, 3, 2, True),    # With communication
        (8, 4, 2, True),    # Large with communication
    ]

    print('\nGenerating scenario GIFs...')
    print('=' * 50)

    for n_agents, n_rooms, capacity, use_comm in scenarios:
        comm_suffix = '_comm' if use_comm else ''
        filename = f'{n_agents}agents_{n_rooms}rooms_cap{capacity}{comm_suffix}.gif'
        generate_scenario_gif(n_agents, n_rooms, capacity, output_dir / filename, use_comm)

    print('\n' + '=' * 50)
    print('All scenario GIFs generated!')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
