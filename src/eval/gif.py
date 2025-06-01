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

import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for file saving
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.distributions.continuous import TanhNormal


def make_gif(env, policy_module, steps=300, gif_path="outputs/mingle.gif", fps=10, device=None):
    """
    Runs a multi-agent environment simulation and saves a GIF of the rollout with enhanced room visualization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    env.to(device)
    td = env._reset()
    frames = []

    for step in range(steps):
        observation = td.select("observation")

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td_action = policy_module(observation)
            action_dist = TanhNormal(td_action["loc"], td_action["scale"])
            actions = action_dist.sample()

        td = env._step(TensorDict({"action": actions}, batch_size=[]))

        rewards = td.get("reward").squeeze(-1).cpu().numpy()
        mean_reward = rewards.mean()

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_aspect('equal')
        ax.set_title(f"Step {step} | Phase: {env.phase} | Mean Reward: {mean_reward:.2f}")

        # Arena bounds
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

        # Center visual
        if env.phase == "spinning":
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightblue", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="blue", linestyle="-"))
        else:
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="red", linestyle="-"))

        # Draw rooms
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

        # Agents
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

        if env.phase == "spinning":
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[~in_center, 0], positions[~in_center, 1], c="red", s=80, label="Outside Center")
        else:
            ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
            ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
            ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside All")

        for i, pos in enumerate(positions):
            ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

        ax.legend(loc='upper right')

        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"🎞️  GIF saved to {gif_path}\n")
    return gif_path


def load_images(image_paths, max_size=50):
    """
    Load and prepare images for animation.

    Parameters:
        image_paths: List of paths to the images
        max_size: Maximum size for the images in pixels

    Returns:
        List of NumPy arrays containing the image data
    """
    images = []

    # Check if we have valid images
    valid_images_found = False

    # Try to load all provided images
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)
            images.append(np.array(img))
            valid_images_found = True
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    # If no valid images were loaded, create placeholder images
    if not valid_images_found:
        print("No valid images found. Creating placeholder images...")
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

        # Create placeholder images
        for i in range(max(len(image_paths), 2)):
            placeholder = np.zeros((max_size, max_size, 4), dtype=np.uint8)
            color_idx = i % len(colors)
            placeholder[:, :, 0] = colors[color_idx][0]  # R
            placeholder[:, :, 1] = colors[color_idx][1]  # G
            placeholder[:, :, 2] = colors[color_idx][2]  # B
            placeholder[:, :, 3] = 255  # Full opacity
            images.append(placeholder)

    return images


def make_gif_char(env, policy_module, steps=300, gif_path="outputs/mingle.gif", fps=10, device=None,
             use_character_animation=False,
             runner_image_paths=None,
             standing_image_paths=None,
             image_size=50,
             speed_threshold=0.05):  # Speed threshold to switch from standing to running
    """
    Runs a multi-agent environment simulation and saves a GIF of the rollout with enhanced room visualization.

    Args:
        env: The environment to run
        policy_module: The policy to use for agent actions
        steps: Number of steps to simulate
        gif_path: Path to save the output GIF
        fps: Frames per second for the GIF
        device: Device to run on (cuda or cpu)
        use_character_animation: Whether to use character animation for agents
        runner_image_paths: List of paths to runner images
        standing_image_paths: List of paths to standing images
        image_size: Size to resize images to
        speed_threshold: Speed threshold to switch between standing and running animations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Load images if requested
    runner_images = None
    standing_images = None
    if use_character_animation:
        if runner_image_paths:
            runner_images = load_images(runner_image_paths, max_size=image_size)
            print(f"Loaded {len(runner_images)} runner images")
        else:
            print("No runner images provided")

        if standing_image_paths:
            standing_images = load_images(standing_image_paths, max_size=image_size)
            print(f"Loaded {len(standing_images)} standing images")
        else:
            print("No standing images provided, will create defaults")
            # Create 2 default standing images if none provided
            standing_images = [
                np.zeros((image_size, image_size, 4), dtype=np.uint8),
                np.zeros((image_size, image_size, 4), dtype=np.uint8)
            ]
            # Make them different colors
            standing_images[0][:, :, 0] = 200  # Red-ish
            standing_images[0][:, :, 3] = 255  # Full opacity
            standing_images[1][:, :, 2] = 200  # Blue-ish
            standing_images[1][:, :, 3] = 255  # Full opacity

    env.to(device)
    td = env._reset()
    frames = []

    # Store previous positions to calculate movement direction and speed
    prev_positions = None
    agent_speeds = None  # Will track the speed of each agent

    for step in range(steps):
        observation = td.select("observation")

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td_action = policy_module(observation)
            action_dist = TanhNormal(td_action["loc"], td_action["scale"])
            actions = action_dist.sample()

        td = env._step(TensorDict({"action": actions}, batch_size=[]))

        rewards = td.get("reward").squeeze(-1).cpu().numpy()
        mean_reward = rewards.mean()

        # Get current positions
        current_positions = env.agent_positions.detach().cpu().numpy()

        # Calculate directions and speeds based on previous positions
        directions = np.zeros((len(current_positions), 2))
        speeds = np.zeros(len(current_positions))

        if prev_positions is not None:
            # Calculate movement vectors
            movement = current_positions - prev_positions
            # Calculate speeds (magnitude of movement)
            speeds = np.linalg.norm(movement, axis=1)
            # Normalize for direction only where there is movement
            for i, (move, speed) in enumerate(zip(movement, speeds)):
                if speed > 1e-6:  # Only normalize if there's significant movement
                    directions[i] = move / speed

        # Update agent_speeds
        agent_speeds = speeds

        # Update previous positions for next frame
        prev_positions = current_positions.copy()

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_ylim(-env.arena_radius - env.room_radius, env.arena_radius + env.room_radius)
        ax.set_aspect('equal')
        ax.set_title(f"Step {step} | Phase: {env.phase} | Mean Reward: {mean_reward:.2f}")

        # Arena bounds
        ax.add_artist(plt.Circle((0, 0), env.arena_radius, fill=False, color="gray", linestyle="--"))

        # Center visual
        if env.phase == "spinning":
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightblue", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="blue", linestyle="-"))
        else:
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=True, color="lightsalmon", alpha=0.3))
            ax.add_artist(plt.Circle((0, 0), env.center_radius, fill=False, color="red", linestyle="-"))

        # Draw rooms
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

        # Draw agents
        if use_character_animation and (runner_images or standing_images):
            # Use character animation for agents
            for i, (pos, dir_vec, speed) in enumerate(zip(positions, directions, agent_speeds)):
                # Determine agent category for coloring
                if env.phase == "spinning":
                    if in_center[i]:
                        color_multiplier = np.array([0.6, 0.6, 1.0, 1.0])  # Blue tint
                    else:
                        color_multiplier = np.array([1.0, 0.6, 0.6, 1.0])  # Red tint
                else:
                    if in_room[i]:
                        color_multiplier = np.array([0.6, 1.0, 0.6, 1.0])  # Green tint
                    elif in_center[i]:
                        color_multiplier = np.array([0.6, 0.6, 1.0, 1.0])  # Blue tint
                    else:
                        color_multiplier = np.array([1.0, 0.8, 0.6, 1.0])  # Orange tint

                # Determine whether to use standing or running animation based on speed
                is_standing = speed < speed_threshold

                if is_standing and standing_images:
                    # Use standing animation - alternate between the two standing images slowly
                    img_index = (step // 10 + i) % len(standing_images)
                    img = standing_images[img_index].copy()
                else:
                    # Use running animation - cycle through images faster for movement
                    img_index = (step + i) % len(runner_images) if runner_images else 0
                    img = runner_images[img_index].copy() if runner_images else standing_images[0].copy()

                # Apply color tint
                img = img.astype(np.float32) / 255.0
                img = (img * color_multiplier).clip(0, 1)
                img = (img * 255).astype(np.uint8)

                # Determine direction for flipping
                facing_left = dir_vec[0] < 0

                # Mirror image if facing left
                if facing_left:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                    img = np.array(pil_img)

                # Create and add annotation box
                im = OffsetImage(img, zoom=0.5)
                ab = AnnotationBbox(im, (pos[0], pos[1]), frameon=False, box_alignment=(0.5, 0.5))
                ax.add_artist(ab)

                # Add agent number
                ax.text(pos[0] + 0.3, pos[1] + 0.3, str(i), fontsize=8, color='black')
        else:
            # Use scatter points for agents (original method)
            if env.phase == "spinning":
                ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
                ax.scatter(positions[~in_center, 0], positions[~in_center, 1], c="red", s=80, label="Outside Center")
            else:
                ax.scatter(positions[in_room, 0], positions[in_room, 1], c="green", s=80, label="In Room")
                ax.scatter(positions[in_center, 0], positions[in_center, 1], c="blue", s=80, label="In Center")
                ax.scatter(positions[outside_all, 0], positions[outside_all, 1], c="orange", s=80, label="Outside All")

            for i, pos in enumerate(positions):
                ax.text(pos[0] + 0.2, pos[1] + 0.2, str(i), fontsize=8, color='black')

        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(image[..., :3])
        plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"🎞️  GIF saved to {gif_path}\n")
    return gif_path