import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tqdm import tqdm
from collections import deque
import numpy as np


class MAPPOActor(nn.Module):
    """MAPPO actor network with layer normalization."""
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        # Deeper network with layer normalization for stability
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size // 2, action_dim)
        # Initialize log_std to a reasonable value
        self.logstd = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        # Clamp std to prevent numerical issues
        std = torch.exp(self.logstd.clamp(-5, 2)).expand_as(mean)
        return mean, std


class CentralValueNetwork(nn.Module):
    """Centralized value network for CTDE paradigm."""
    def __init__(self, obs_dim, n_agents, hidden_size=256):
        super().__init__()
        input_dim = obs_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, obs_all_agents):
        if obs_all_agents.dim() == 2:
            obs_all_agents = obs_all_agents.unsqueeze(0)

        batch, n, d = obs_all_agents.shape
        flat = obs_all_agents.reshape(batch, n * d)
        return self.net(flat)


class TrajectoryBuffer:
    """Buffer for collecting trajectories before updates."""

    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self):
        return len(self.observations)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_batch(self, device):
        """Convert buffer to tensors for training."""
        obs = torch.stack(self.observations).to(device)
        actions = torch.stack(self.actions).to(device)
        rewards = torch.stack(self.rewards).to(device)
        dones = torch.stack(self.dones).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        values = torch.stack(self.values).to(device)
        return obs, actions, rewards, dones, log_probs, values


class MAPPOTrainer:
    """Multi-Agent PPO with Centralized Training, Decentralized Execution (CTDE)."""

    def __init__(self, env, device="cpu", config=None):
        self.env = env
        self.device = device

        # Extract shapes - handle communication environment with multiple action keys
        full_action_spec = env.full_action_spec_unbatched
        if "action" in full_action_spec.keys():
            action_spec = full_action_spec["action"]
        else:
            # Fallback to single action key
            action_spec = full_action_spec[env.action_key]
        obs_spec = env.observation_spec["observation"]

        self.n_agents = action_spec.shape[0]
        self.action_dim = action_spec.shape[-1]
        self.obs_dim = obs_spec.shape[-1]

        # Check if environment has communication (message actions)
        self.has_communication = "message" in full_action_spec.keys()

        # Hyperparameters (with better defaults)
        config = config or {}
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 2048)
        self.minibatch_size = config.get("minibatch_size", 256)
        self.num_epochs = config.get("num_epochs", 10)
        self.lr = config.get("lr", 3e-4)

        # Networks (larger for multi-agent)
        hidden_size = config.get("hidden_size", 256)
        self.actor = MAPPOActor(self.obs_dim, self.action_dim, hidden_size).to(device)
        self.critic = CentralValueNetwork(self.obs_dim, self.n_agents, hidden_size).to(device)

        # Single optimizer for both networks (often works better)
        self.optimizer = optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            eps=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(0.1, 1.0 - step / 1000)
        )

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(self.batch_size)

    @torch.no_grad()
    def select_action(self, obs):
        """Select action and compute log probability (no gradients)."""
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        # Compute log prob with tanh correction
        action_clamped = action.clamp(-0.999, 0.999)
        raw_action_recovered = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))
        log_prob = dist.log_prob(raw_action_recovered).sum(-1)
        # Tanh correction for log prob
        log_prob -= torch.log(1 - action_clamped.pow(2) + 1e-6).sum(-1)

        return action, log_prob

    @torch.no_grad()
    def compute_value(self, obs):
        """Compute centralized value estimate."""
        return self.critic(obs)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (~dones[t]).float() - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (~dones[t]).float() * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_log_probs, advantages, returns):
        """Perform PPO update on collected batch."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_epoch = 0
        num_updates = 0

        # Multiple epochs over the batch
        for _ in range(self.num_epochs):
            # Generate random minibatch indices
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.minibatch_size):
                end = min(start + self.minibatch_size, len(obs))
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass - handle (batch, n_agents, obs_dim) shape
                # Flatten batch for actor: (batch * n_agents, obs_dim)
                batch_size = mb_obs.shape[0]
                if mb_obs.dim() == 3:
                    # (batch, n_agents, obs_dim) -> (batch * n_agents, obs_dim)
                    n_agents_batch = mb_obs.shape[1]
                    obs_flat = mb_obs.reshape(-1, mb_obs.shape[-1])
                    actions_flat = mb_actions.reshape(-1, mb_actions.shape[-1])
                else:
                    obs_flat = mb_obs
                    actions_flat = mb_actions
                    n_agents_batch = self.n_agents

                mean, std = self.actor(obs_flat)
                dist = torch.distributions.Normal(mean, std)

                # Compute new log probs
                action_clamped = actions_flat.clamp(-0.999, 0.999)
                raw_action = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))
                new_log_probs = dist.log_prob(raw_action).sum(-1)
                new_log_probs -= torch.log(1 - action_clamped.pow(2) + 1e-6).sum(-1)

                # Reshape and average over agents to match stored log_probs
                if mb_obs.dim() == 3:
                    new_log_probs = new_log_probs.reshape(batch_size, n_agents_batch).mean(dim=1)

                # Entropy
                entropy = dist.entropy().sum(-1).mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.critic(mb_obs).squeeze()
                value_loss = 0.5 * (values - mb_returns).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_loss_epoch += loss.item()
                num_updates += 1

        self.scheduler.step()
        return total_loss_epoch / max(num_updates, 1)

    def train_mappo(self, total_frames, log_interval=100):
        """Main training loop with batch collection."""
        td = self.env.reset()
        obs = td["observation"].detach().clone()

        logs = {
            "episode_rewards": [],
            "losses": [],
            "frames": [],
        }
        episode_reward = 0.0
        episode_count = 0
        recent_rewards = deque(maxlen=100)

        pbar = tqdm(total=total_frames, desc="MAPPO Training")
        frame = 0

        while frame < total_frames:
            # ============ COLLECT TRAJECTORY BATCH ============
            self.buffer.clear()

            for _ in range(self.batch_size // self.n_agents):
                # Select action
                action, log_prob = self.select_action(obs)
                value = self.compute_value(obs)

                # Environment step - include message if communication is enabled
                step_dict = {"action": action}
                if self.has_communication:
                    # Generate random message for now (MAPPO doesn't learn messages)
                    step_dict["message"] = torch.randint(0, 2, (self.n_agents,), device=self.device)
                step_td = TensorDict(step_dict, batch_size=[])
                td_next = self.env.step(step_td)
                next_td = td_next["next"]

                next_obs = next_td["observation"].detach().clone()
                reward = next_td["reward"].detach().clone()
                done = next_td["done"].detach().clone()

                # Store transition (aggregate across agents)
                self.buffer.add(
                    obs.detach().clone(),
                    action.detach().clone(),
                    reward.sum().detach().clone(),  # Team reward
                    done.any().detach().clone(),
                    log_prob.mean().detach().clone(),  # Average log prob
                    value.squeeze().detach().clone()
                )

                episode_reward += reward.sum().item()
                frame += self.n_agents
                pbar.update(self.n_agents)

                # Handle episode end
                if done.any():
                    logs["episode_rewards"].append(episode_reward)
                    recent_rewards.append(episode_reward)
                    episode_count += 1
                    episode_reward = 0.0
                    td = self.env.reset()
                    obs = td["observation"].detach().clone()
                else:
                    obs = next_obs

                if frame >= total_frames:
                    break

            # ============ UPDATE NETWORKS ============
            if len(self.buffer) > self.minibatch_size:
                # Get batch data
                batch_obs, batch_actions, batch_rewards, batch_dones, batch_log_probs, batch_values = \
                    self.buffer.get_batch(self.device)

                # Compute next value for GAE
                with torch.no_grad():
                    next_value = self.compute_value(obs).squeeze()

                # Compute advantages and returns
                advantages, returns = self.compute_gae(
                    batch_rewards, batch_values, batch_dones, next_value
                )

                # Perform PPO update
                loss = self.update(
                    batch_obs, batch_actions, batch_log_probs,
                    advantages, returns
                )

                logs["losses"].append(loss)
                logs["frames"].append(frame)

            # Logging
            if episode_count % log_interval == 0 and episode_count > 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"\n[MAPPO] Frame {frame}/{total_frames} | "
                      f"Episodes: {episode_count} | "
                      f"Avg Reward (100): {avg_reward:.2f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        pbar.close()
        return logs


def train(
    collector=None,
    loss_module=None,
    advantage_module=None,
    replay_buffer=None,
    optim=None,
    device=None,
    total_frames=2000,
    frames_per_batch=None,
    num_epochs=None,
    minibatch_size=None,
    max_grad_norm=None,
    env=None,
    scheduler=None,
    log_interval=None,
    metrics_save_path="mappo_metrics.json",
    gif_interval=None,
    policy_module=None,
    eval_episodes=None,
    use_character_animation=True,
):

    trainer = MAPPOTrainer(env, device or "cpu")
    
    try:
        logs = trainer.train_mappo(total_frames=total_frames)
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        logs = {"episode_rewards": [], "error": str(e)}

    # Save logs
    try:
        import json
        with open(metrics_save_path, "w") as f:
            json.dump(logs, f)
        print(f"Metrics saved to {metrics_save_path}")
    except Exception as e:
        print(f"Failed to save MAPPO metrics: {e}")

    return logs