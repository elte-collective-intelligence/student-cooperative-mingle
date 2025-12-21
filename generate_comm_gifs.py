"""Generate GIFs for discrete and continuous communication trained models."""
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch.nn as nn
from tensordict import TensorDict
from src.envs.modules.reward_module import (
    CollisionAvoidanceReward, StayInRoomReward, GetToRoomReward
)

# Config
n_agents = 6
n_rooms = 3
vocab_size = 8
embed_dim = 16
device = torch.device("cpu")

# Create reward modules - ROOM IS KING! Small penalties so agents can explore.
reward_modules = [
    CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="claiming"),
    GetToRoomReward(max_reward=3.0, phase_mode="claiming"),
    StayInRoomReward(max_reward=10.0, outside_penalty=0.1, overfill_penalty=1.0, phase_mode="claiming"),
]
for m in reward_modules:
    m._activate()

# ========== DISCRETE COMMUNICATION GIF ==========
print("Generating Discrete Communication GIF...")
from communication_channel.envs.discrete_comm_env import MingleEnvWithComm

env = MingleEnvWithComm(
    n_agents=n_agents, n_rooms=n_rooms, arena_radius=10.0, center_radius=3.0,
    max_steps=300, room_radius=2.0, room_capacity=2, vocab_size=vocab_size,
    comm_range=None, reward_modules=reward_modules, phase_mode="claiming"
)

obs_dim = env.observation_spec["observation"].shape[-1]

class CommPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.action_mean = nn.Linear(128, 2)
        self.action_std = nn.Linear(128, 2)
        self.message = nn.Linear(128, vocab_size)

    def forward(self, obs):
        if obs.dim() == 3:
            b, n, d = obs.shape
            obs = obs.view(-1, d)
            h = self.net(obs)
            mean = self.action_mean(h).view(b, n, -1)
            std = torch.exp(self.action_std(h)).clamp(0.1, 1.0).view(b, n, -1)
            msg = self.message(h).view(b, n, -1)
        else:
            h = self.net(obs)
            mean = self.action_mean(h)
            std = torch.exp(self.action_std(h)).clamp(0.1, 1.0)
            msg = self.message(h)
        return mean, std, msg

policy = CommPolicy().to(device)
ckpt = torch.load("contribution_tests_and_comparisions/communication/discrete/model.pt", map_location=device)
policy.load_state_dict(ckpt["policy"])
policy.eval()

gif_frames = []
td = env.reset()
obs = td["observation"].to(device)

for step in range(300):
    with torch.no_grad():
        mean, std, msg_logits = policy(obs.unsqueeze(0))
        mean, std = mean.squeeze(0), std.squeeze(0)
        msg_logits = msg_logits.squeeze(0)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(-0.5, 0.5)
        msg_dist = torch.distributions.Categorical(logits=msg_logits)
        messages = msg_dist.sample()

    step_td = TensorDict({"action": action.cpu(), "message": messages.cpu()}, batch_size=[])
    next_td = env._step(step_td)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))
    ax.add_patch(Circle((0, 0), env.center_radius, fill=False, color="blue", linestyle=":"))

    colors = ["lightblue", "lightgreen", "lightyellow"]
    if hasattr(env, "room_positions") and env.room_positions is not None:
        for i, rp in enumerate(env.room_positions):
            ax.add_patch(Circle((rp[0].item(), rp[1].item()), env.room_radius, fill=True, alpha=0.3, color=colors[i % 3]))

    pos = env.agent_positions.cpu().numpy()
    agent_colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    for i in range(n_agents):
        ax.plot(pos[i, 0], pos[i, 1], "o", markersize=12, color=agent_colors[i])
        ax.annotate(f"M{messages[i].item()}", (pos[i, 0], pos[i, 1] + 0.5), ha="center", fontsize=8)

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(f"Discrete Comm - Step {step}, Phase: {env.phase}")

    fig.canvas.draw()
    frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    gif_frames.append(frame)
    plt.close(fig)

    if next_td["done"].any() or next_td["terminated"].any():
        break
    obs = next_td["observation"].to(device)

imageio.mimsave("contribution_tests_and_comparisions/communication/discrete/trained_policy.gif", gif_frames, fps=10, loop=0)
print("Discrete GIF saved!")

# ========== CONTINUOUS COMMUNICATION GIF ==========
print("Generating Continuous Communication GIF...")
from communication_channel.envs.continuous_comm_env import MingleEnvWithEmbeddings

# Re-create reward modules - ROOM IS KING! Small penalties.
reward_modules2 = [
    CollisionAvoidanceReward(min_distance=0.5, penalty=0.5, phase_mode="claiming"),
    GetToRoomReward(max_reward=3.0, phase_mode="claiming"),
    StayInRoomReward(max_reward=10.0, outside_penalty=0.1, overfill_penalty=1.0, phase_mode="claiming"),
]
for m in reward_modules2:
    m._activate()

env2 = MingleEnvWithEmbeddings(
    n_agents=n_agents, n_rooms=n_rooms, arena_radius=10.0, center_radius=3.0,
    max_steps=300, room_radius=2.0, room_capacity=2, embedding_dim=embed_dim,
    reward_modules=reward_modules2, phase_mode="claiming"
)

obs_dim2 = env2.observation_spec["observation"].shape[-1]

class ContPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.action_mean = nn.Linear(128, 2)
        self.action_std = nn.Linear(128, 2)
        self.embedding = nn.Linear(128, embed_dim)

    def forward(self, obs):
        if obs.dim() == 3:
            b, n, d = obs.shape
            obs = obs.view(-1, d)
            h = self.net(obs)
            mean = self.action_mean(h).view(b, n, -1)
            std = torch.exp(self.action_std(h)).clamp(0.1, 1.0).view(b, n, -1)
            emb = self.embedding(h).view(b, n, -1)
        else:
            h = self.net(obs)
            mean = self.action_mean(h)
            std = torch.exp(self.action_std(h)).clamp(0.1, 1.0)
            emb = self.embedding(h)
        return mean, std, emb

policy2 = ContPolicy().to(device)
ckpt2 = torch.load("contribution_tests_and_comparisions/communication/continuous/model.pt", map_location=device)
policy2.load_state_dict(ckpt2["policy"])
policy2.eval()

gif_frames2 = []
td2 = env2.reset()
obs2 = td2["observation"].to(device)

for step in range(300):
    with torch.no_grad():
        mean, std, emb = policy2(obs2.unsqueeze(0))
        mean, std = mean.squeeze(0), std.squeeze(0)
        emb = emb.squeeze(0)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(-0.5, 0.5)

    step_td = TensorDict({"action": action.cpu(), "embedding": emb.cpu()}, batch_size=[])
    next_td = env2._step(step_td)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Circle((0, 0), env2.arena_radius, fill=False, color="gray", linestyle="--"))
    ax.add_patch(Circle((0, 0), env2.center_radius, fill=False, color="blue", linestyle=":"))

    if hasattr(env2, "room_positions") and env2.room_positions is not None:
        for i, rp in enumerate(env2.room_positions):
            ax.add_patch(Circle((rp[0].item(), rp[1].item()), env2.room_radius, fill=True, alpha=0.3, color=colors[i % 3]))

    pos = env2.agent_positions.cpu().numpy()
    for i in range(n_agents):
        ax.plot(pos[i, 0], pos[i, 1], "o", markersize=12, color=agent_colors[i])

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(f"Continuous Comm - Step {step}, Phase: {env2.phase}")

    fig.canvas.draw()
    frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    gif_frames2.append(frame)
    plt.close(fig)

    if next_td["done"].any() or next_td["terminated"].any():
        break
    obs2 = next_td["observation"].to(device)

imageio.mimsave("contribution_tests_and_comparisions/communication/continuous/trained_policy.gif", gif_frames2, fps=10, loop=0)
print("Continuous GIF saved!")
print("All communication GIFs generated!")
