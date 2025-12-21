"""
Analyzer for discrete symbolic messages.

Tracks message frequency, transitions, correlations with rewards,
and other emergent communication patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def analyze_discrete_messages(env, policy, device, num_episodes=100, save_dir="communication_results/analysis"):
    """
    Analyze discrete message usage patterns.

    Args:
        env: Communication environment (MingleEnvWithComm)
        policy: Trained policy network
        device: Torch device
        num_episodes: Number of episodes to analyze
        save_dir: Directory to save analysis plots

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(save_dir, exist_ok=True)

    vocab_size = env.vocab_size
    message_counts = torch.zeros(vocab_size)
    message_transitions = torch.zeros(vocab_size, vocab_size)
    message_rewards = defaultdict(list)
    message_phase_usage = {"spinning": torch.zeros(vocab_size), "claiming": torch.zeros(vocab_size)}

    policy.eval()

    print(f"\nAnalyzing {num_episodes} episodes...")

    for episode in range(num_episodes):
        td = env.reset()
        done = False
        prev_messages = None
        step_count = 0

        while not done:
            with torch.no_grad():
                td = policy(td.to(device))

            messages = td["message"].cpu()

            # Count message usage
            for msg in messages:
                message_counts[msg] += 1
                message_phase_usage[env.phase][msg] += 1

            # Track transitions
            if prev_messages is not None:
                for prev_msg, curr_msg in zip(prev_messages, messages):
                    message_transitions[prev_msg, curr_msg] += 1

            # Execute action
            td = env.step(td)
            reward = td["reward"].cpu()

            # Associate messages with rewards
            for msg, r in zip(messages, reward):
                message_rewards[msg.item()].append(r.item())

            done = td["terminated"].any().item()
            prev_messages = messages
            step_count += 1

        if (episode + 1) % 20 == 0:
            print(f"  Analyzed {episode + 1}/{num_episodes} episodes")

    print("Analysis complete. Generating visualizations...")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Message frequency
    axes[0, 0].bar(range(vocab_size), message_counts.numpy())
    axes[0, 0].set_title('Message Frequency', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Message ID')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Message transition matrix
    transition_norm = message_transitions / (message_transitions.sum(dim=1, keepdim=True) + 1e-8)
    im = axes[0, 1].imshow(transition_norm.numpy(), cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Message Transition Probabilities', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Next Message')
    axes[0, 1].set_ylabel('Current Message')
    plt.colorbar(im, ax=axes[0, 1])

    # 3. Reward by message
    msg_reward_means = [np.mean(message_rewards[i]) if message_rewards[i] else 0
                        for i in range(vocab_size)]
    msg_reward_stds = [np.std(message_rewards[i]) if message_rewards[i] else 0
                       for i in range(vocab_size)]

    axes[0, 2].bar(range(vocab_size), msg_reward_means, yerr=msg_reward_stds, capsize=5)
    axes[0, 2].set_title('Average Reward by Message', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Message ID')
    axes[0, 2].set_ylabel('Avg Reward')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. Message entropy (diversity)
    msg_probs = message_counts / message_counts.sum()
    entropy = -torch.sum(msg_probs * torch.log(msg_probs + 1e-8)).item()
    max_entropy = np.log(vocab_size)

    axes[1, 0].bar(['Actual', 'Maximum'], [entropy, max_entropy])
    axes[1, 0].set_title('Message Entropy (Diversity)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0, entropy + 0.1, f'{entropy:.3f}', ha='center', fontsize=12)
    axes[1, 0].text(1, max_entropy + 0.1, f'{max_entropy:.3f}', ha='center', fontsize=12)

    # 5. Phase-specific usage
    spinning_probs = message_phase_usage["spinning"] / (message_phase_usage["spinning"].sum() + 1e-8)
    claiming_probs = message_phase_usage["claiming"] / (message_phase_usage["claiming"].sum() + 1e-8)

    x = np.arange(vocab_size)
    width = 0.35
    axes[1, 1].bar(x - width/2, spinning_probs.numpy(), width, label='Spinning', alpha=0.8)
    axes[1, 1].bar(x + width/2, claiming_probs.numpy(), width, label='Claiming', alpha=0.8)
    axes[1, 1].set_title('Message Usage by Phase', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Message ID')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Message specialization (reward variance)
    msg_reward_vars = [np.var(message_rewards[i]) if len(message_rewards[i]) > 1 else 0
                       for i in range(vocab_size)]
    axes[1, 2].bar(range(vocab_size), msg_reward_vars)
    axes[1, 2].set_title('Message Context Variance', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Message ID')
    axes[1, 2].set_ylabel('Reward Variance')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'message_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved analysis plot to: {plot_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("  MESSAGE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nMessage Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"Entropy Ratio: {entropy/max_entropy:.1%}")
    print(f"\nMost Used Messages:")
    top_3 = torch.argsort(message_counts, descending=True)[:3]
    for i, msg_id in enumerate(top_3):
        pct = (message_counts[msg_id] / message_counts.sum() * 100).item()
        print(f"  {i+1}. Message {msg_id}: {message_counts[msg_id].item():.0f} times ({pct:.1f}%)")

    print(f"\nBest Rewarded Messages:")
    sorted_by_reward = sorted(range(vocab_size), key=lambda i: np.mean(message_rewards[i]) if message_rewards[i] else float('-inf'), reverse=True)[:3]
    for i, msg_id in enumerate(sorted_by_reward):
        if message_rewards[msg_id]:
            avg_reward = np.mean(message_rewards[msg_id])
            print(f"  {i+1}. Message {msg_id}: Avg Reward = {avg_reward:.3f}")

    return {
        'counts': message_counts,
        'transitions': message_transitions,
        'rewards_by_message': dict(message_rewards),
        'entropy': entropy,
        'max_entropy': max_entropy,
        'phase_usage': message_phase_usage,
        'plot_path': plot_path
    }
