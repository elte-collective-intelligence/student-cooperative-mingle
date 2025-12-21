import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for file saving
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import imageio
from PIL import Image
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.distributions.continuous import TanhNormal


def make_gif(env, policy_module, steps=300, gif_path="outputs/mingle.gif", fps=10, device=None):
    """
    Runs a multi-agent environment simulation and saves a GIF of the rollout.
    RESTORED to original v1 visualization style.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    env.to(device)
    td = env._reset()
    frames = []

    # Check if this is a communication-enabled environment
    has_comm = hasattr(env, 'vocab_size') and hasattr(env, 'message_buffer')
    if has_comm:
        print(f"[GIF] Communication visualization enabled (vocab_size={env.vocab_size})")

    for step in range(steps):
        observation = td.select("observation")

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td_action = policy_module(observation)

            # Handle both continuous (TanhNormal) and discrete action spaces
            if "loc" in td_action.keys() and "scale" in td_action.keys():
                action_dist = TanhNormal(td_action["loc"], td_action["scale"])
                actions = action_dist.sample()
            else:
                actions = td_action.get("action", td_action.get("loc", None))

            # Handle message output for communication environments
            if has_comm and "message" in td_action.keys():
                messages = td_action["message"]
                td_step = TensorDict({"action": actions, "message": messages}, batch_size=[])
            else:
                td_step = TensorDict({"action": actions}, batch_size=[])

        td = env._step(td_step)

        rewards = td.get("reward").squeeze(-1).cpu().numpy()
        mean_reward = rewards.mean()

        # Plotting - ORIGINAL v1 STYLE
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_aspect('equal')
        ax.set_title(f"Step {step} | Phase: {env.phase} | Mean Reward: {mean_reward:.2f}")

        # Arena bounds - ORIGINAL: gray dashed
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

        # Center visual - ORIGINAL style
        if env.phase == "spinning":
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightblue", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="blue", linestyle="-"))
        else:
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="red", linestyle="-"))

        # Draw rooms - ORIGINAL style
        if hasattr(env, "room_positions") and env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()

            if hasattr(env, "room_occupancy"):
                room_occupancies = env.room_occupancy.cpu().numpy()
            else:
                room_occupancies = np.zeros(len(room_positions))

            for i, (room_pos, occupancy) in enumerate(zip(room_positions, room_occupancies)):
                if hasattr(env, "room_capacity"):
                    fill_ratio = min(occupancy / env.room_capacity, 1.0)
                    if fill_ratio == 1.0:
                        color = "green"
                        alpha = 0.4
                    elif fill_ratio < 1.0:
                        color = "yellow"
                        alpha = 0.3 * fill_ratio
                    else:
                        color = "red"
                        alpha = 0.3
                else:
                    color = "green"
                    alpha = 0.2

                ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=True, color=color, alpha=alpha))
                ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=False, color="green", linestyle="-"))

                if hasattr(env, "room_capacity"):
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occupancy)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10)
                else:
                    ax.text(room_pos[0], room_pos[1], f"Room {i}", ha='center', va='center', fontsize=8)

        # Agents - ORIGINAL style
        positions = env.agent_positions.detach().cpu().numpy()

        in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
        in_room = np.zeros(len(positions), dtype=bool)
        room_assignments = np.full(len(positions), -1)

        if hasattr(env, "room_positions") and env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()
            for i, pos in enumerate(positions):
                distances = np.linalg.norm(room_positions - pos, axis=1)
                closest_room = np.argmin(distances)
                if distances[closest_room] < env.room_radius:
                    in_room[i] = True
                    room_assignments[i] = closest_room

        outside_all = ~in_center & ~in_room

        # ORIGINAL scatter plot style
        if env.phase == "spinning":
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[~in_center, 0], positions[~in_center, 1], c="red", s=80, label="Outside Center")
        else:
            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside All")

        # Agent labels with message visualization
        if has_comm and hasattr(env, 'message_buffer'):
            messages = env.message_buffer.cpu().numpy()
            # Message visualization:
            # Message 0 = "Follow me" (green arrow up) - shown for leaders in spinning, all in claiming
            # Message 1 = "Room full" (red X) - only shown in claiming phase when room is full
            msg_colors = ['#2ecc71', '#e74c3c']  # Green, Red
            msg_symbols = ['^', 'X']  # Arrow up, X
            msg_labels = ['Follow me', 'Room full']

            # Get leaders during spinning phase
            spinning_leaders = getattr(env, '_spinning_leaders', None)

            for i, pos in enumerate(positions):
                msg_idx = int(messages[i]) if i < len(messages) else 0

                # During spinning phase: only leaders show "Follow me"
                if env.phase == "spinning":
                    if spinning_leaders is not None and i in spinning_leaders:
                        # Leader - show "Follow me"
                        ax.scatter(pos[0], pos[1] + 0.6, marker=msg_symbols[0],
                                  c=msg_colors[0], s=120, zorder=10, edgecolors='black', linewidths=1)
                    # Non-leaders: no message marker during spinning
                else:
                    # Claiming phase: show actual message based on room state
                    ax.scatter(pos[0], pos[1] + 0.6, marker=msg_symbols[msg_idx],
                              c=msg_colors[msg_idx], s=100, zorder=10, edgecolors='black', linewidths=0.5)

                # Agent number
                ax.text(pos[0] + 0.3, pos[1] - 0.3, str(i), fontsize=8, color='black')

            # Add message legend
            for idx, (sym, col, lbl) in enumerate(zip(msg_symbols, msg_colors, msg_labels)):
                ax.scatter([], [], marker=sym, c=col, s=80, label=f"Msg: {lbl}", edgecolors='black', linewidths=0.5)
        else:
            # No communication - just show agent numbers
            for i, pos in enumerate(positions):
                ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

        ax.legend(loc='upper right', fontsize=8)

        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[GIF] GIF saved to {gif_path}")
    return gif_path


def make_gif_char(env, policy_module, steps=300, gif_path="outputs/mingle.gif", fps=10, device=None,
             use_character_animation=False,
             runner_image_paths=None,
             standing_image_paths=None,
             image_size=50,
             speed_threshold=0.05):
    """
    Runs a multi-agent environment simulation with optional character animation.
    RESTORED to original v1 style.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Check if this is a communication-enabled environment
    has_comm = hasattr(env, 'vocab_size') and hasattr(env, 'message_buffer')
    if has_comm:
        print(f"[GIF] Communication visualization enabled (vocab_size={env.vocab_size})")

    env.to(device)
    td = env._reset()
    frames = []

    prev_positions = None
    agent_speeds = None

    for step in range(steps):
        observation = td.select("observation")

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td_action = policy_module(observation)

            if "loc" in td_action.keys() and "scale" in td_action.keys():
                action_dist = TanhNormal(td_action["loc"], td_action["scale"])
                actions = action_dist.sample()
            else:
                actions = td_action.get("action", td_action.get("loc", None))

            if has_comm and "message" in td_action.keys():
                messages = td_action["message"]
                td_step = TensorDict({"action": actions, "message": messages}, batch_size=[])
            else:
                td_step = TensorDict({"action": actions}, batch_size=[])

        td = env._step(td_step)

        rewards = td.get("reward").squeeze(-1).cpu().numpy()
        mean_reward = rewards.mean()

        current_positions = env.agent_positions.detach().cpu().numpy()

        directions = np.zeros((len(current_positions), 2))
        speeds = np.zeros(len(current_positions))

        if prev_positions is not None:
            movement = current_positions - prev_positions
            speeds = np.linalg.norm(movement, axis=1)
            for i, (move, speed) in enumerate(zip(movement, speeds)):
                if speed > 1e-6:
                    directions[i] = move / speed

        agent_speeds = speeds
        prev_positions = current_positions.copy()

        # Plotting - ORIGINAL v1 STYLE
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_aspect('equal')
        ax.set_title(f"Step {step} | Phase: {env.phase} | Mean Reward: {mean_reward:.2f}")

        # Arena bounds - ORIGINAL
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

        # Center visual - ORIGINAL
        if env.phase == "spinning":
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightblue", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="blue", linestyle="-"))
        else:
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="red", linestyle="-"))

        # Draw rooms - ORIGINAL
        if hasattr(env, "room_positions") and env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()

            if hasattr(env, "room_occupancy"):
                room_occupancies = env.room_occupancy.cpu().numpy()
            else:
                room_occupancies = np.zeros(len(room_positions))

            for i, (room_pos, occupancy) in enumerate(zip(room_positions, room_occupancies)):
                if hasattr(env, "room_capacity"):
                    fill_ratio = min(occupancy / env.room_capacity, 1.0)
                    if fill_ratio == 1.0:
                        color = "green"
                        alpha = 0.4
                    elif fill_ratio < 1.0:
                        color = "yellow"
                        alpha = 0.3 * fill_ratio
                    else:
                        color = "red"
                        alpha = 0.3
                else:
                    color = "green"
                    alpha = 0.2

                ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=True, color=color, alpha=alpha))
                ax.add_artist(plt.Circle(room_pos, radius=env.room_radius, fill=False, color="green", linestyle="-"))

                if hasattr(env, "room_capacity"):
                    ax.text(room_pos[0], room_pos[1], f"R{i}: {int(occupancy)}/{env.room_capacity}",
                            ha='center', va='center', fontsize=10)
                else:
                    ax.text(room_pos[0], room_pos[1], f"Room {i}", ha='center', va='center', fontsize=8)

        # Classify agents
        positions = current_positions
        in_center = np.linalg.norm(positions, axis=1) <= env.center_radius
        in_room = np.zeros(len(positions), dtype=bool)
        room_assignments = np.full(len(positions), -1)

        if hasattr(env, "room_positions") and env.room_positions is not None:
            room_positions = env.room_positions.cpu().numpy()
            for i, pos in enumerate(positions):
                distances = np.linalg.norm(room_positions - pos, axis=1)
                closest_room = np.argmin(distances)
                if distances[closest_room] < env.room_radius:
                    in_room[i] = True
                    room_assignments[i] = closest_room

        outside_all = ~in_center & ~in_room

        # ORIGINAL scatter plot style (no character animation for simplicity)
        if env.phase == "spinning":
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[~in_center, 0], positions[~in_center, 1], c="red", s=80, label="Outside Center")
        else:
            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside All")

        # Agent labels with message visualization
        if has_comm and hasattr(env, 'message_buffer'):
            messages = env.message_buffer.cpu().numpy()
            # Message visualization:
            # Message 0 = "Follow me" (green arrow up) - shown for leaders in spinning, all in claiming
            # Message 1 = "Room full" (red X) - only shown in claiming phase when room is full
            msg_colors = ['#2ecc71', '#e74c3c']  # Green, Red
            msg_symbols = ['^', 'X']  # Arrow up, X
            msg_labels = ['Follow me', 'Room full']

            # Get leaders during spinning phase
            spinning_leaders = getattr(env, '_spinning_leaders', None)

            for i, pos in enumerate(positions):
                msg_idx = int(messages[i]) if i < len(messages) else 0

                # During spinning phase: only leaders show "Follow me"
                if env.phase == "spinning":
                    if spinning_leaders is not None and i in spinning_leaders:
                        # Leader - show "Follow me"
                        ax.scatter(pos[0], pos[1] + 0.6, marker=msg_symbols[0],
                                  c=msg_colors[0], s=120, zorder=10, edgecolors='black', linewidths=1)
                    # Non-leaders: no message marker during spinning
                else:
                    # Claiming phase: show actual message based on room state
                    ax.scatter(pos[0], pos[1] + 0.6, marker=msg_symbols[msg_idx],
                              c=msg_colors[msg_idx], s=100, zorder=10, edgecolors='black', linewidths=0.5)

                # Agent number
                ax.text(pos[0] + 0.3, pos[1] - 0.3, str(i), fontsize=8, color='black')

            # Add message legend
            for idx, (sym, col, lbl) in enumerate(zip(msg_symbols, msg_colors, msg_labels)):
                ax.scatter([], [], marker=sym, c=col, s=80, label=f"Msg: {lbl}", edgecolors='black', linewidths=0.5)
        else:
            # No communication - just show agent numbers
            for i, pos in enumerate(positions):
                ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

        ax.legend(loc='upper right', fontsize=8)

        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[GIF] GIF saved to {gif_path}")
    return gif_path
