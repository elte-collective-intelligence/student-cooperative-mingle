import torch
import math
from typing import List, Optional, Tuple
from tensordict import TensorDict
from torchrl.data import Composite, Bounded, Unbounded, Categorical

from src.envs.mingle_env import MingleEnv
from src.envs.modules.reward_module import RewardModule


# Simple 2-message vocabulary with clear meaning
MSG_FOLLOW_ME = 0    # "Follow me" - I'm heading to a room
MSG_ROOM_FULL = 1    # "Room is full" - Don't come here


class MingleEnvWithComm(MingleEnv):
    """
    MingleEnv extended with simple 2-message communication.

    Messages:
        0 = "Follow me" - Agent is heading to a room, others can follow
        1 = "Room is full" - Agent's room is full, find another

    Features:
        - Agents can move freely (no freezing)
        - Room tracking for reward computation
        - Simple messages make learning easier than arbitrary vocabulary
        - Agent IDs for symmetry breaking

    Parameters:
        vocab_size (int): Number of discrete messages (default: 2)
        comm_range (float, optional): Communication range. None = global broadcast.
        include_agent_id (bool): Whether to include one-hot agent ID in observation
    """

    def __init__(
        self,
        n_agents: int = 2,
        n_rooms: int = 2,
        arena_radius: float = 10.0,
        center_radius: float = 3.0,
        max_steps: int = 300,
        spinning_phase_range: Tuple[int, int] = (50, 100),
        room_radius: float = 3.0,
        room_capacity: int = 2,
        reward_modules: Optional[List[RewardModule]] = None,
        reward_managers: Optional[dict] = None,
        phase_mode: str = "both",
        vocab_size: int = 2,  # Changed: 2 messages ("Follow me", "Room full")
        comm_range: Optional[float] = None,
        include_agent_id: bool = True,
    ) -> None:
        # Initialize parent first
        super().__init__(
            n_agents=n_agents,
            n_rooms=n_rooms,
            arena_radius=arena_radius,
            center_radius=center_radius,
            max_steps=max_steps,
            spinning_phase_range=spinning_phase_range,
            room_radius=room_radius,
            room_capacity=room_capacity,
            reward_modules=reward_modules,
            reward_managers=reward_managers,
            phase_mode=phase_mode
        )

        self.vocab_size = vocab_size
        self.comm_range = comm_range
        self.include_agent_id = include_agent_id
        self.message_buffer = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

        # Room tracking (no freezing - agents can move freely)
        self.agent_in_room = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        # === NEW: Advanced tracking for fair rewards ===
        # Track previous room index (for persistence bonus)
        self.prev_agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        # Track consecutive steps in same room
        self.room_persistence_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        # Track room entry order: for each room, list of (agent_idx, entry_step)
        # Using a dict: room_idx -> list of agent indices in entry order
        self.room_entry_order = {i: [] for i in range(n_rooms)}
        # Track global step counter for entry ordering
        self.global_step = 0
        # Track which agents received "Room Full" warning from agents in each room
        # received_room_full_from[agent_i] = set of room indices where warning came from
        self.received_room_full_warning = {i: set() for i in range(n_agents)}

        # Calculate observation dimension:
        # Base obs: 14
        # + messages from other agents (one-hot encoded): (n_agents - 1) * vocab_size
        # + agent ID (one-hot encoded): n_agents (if include_agent_id)
        messages_obs_dim = (self.n_agents - 1) * self.vocab_size
        agent_id_dim = self.n_agents if self.include_agent_id else 0
        new_obs_dim = 14 + messages_obs_dim + agent_id_dim

        # Store for reference
        self._base_obs_dim = 14
        self._messages_obs_dim = messages_obs_dim
        self._agent_id_dim = agent_id_dim

        # Update observation spec
        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, new_obs_dim), device=self.device),
        })

        # Update action spec to include message output
        self.action_spec = Composite({
            "action": Bounded(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(self.n_agents, 2),
                device=self.device,
            ),
            "message": Categorical(
                n=self.vocab_size,
                shape=(self.n_agents,),
                device=self.device,
            ),
        })

    def _get_agent_ids(self) -> torch.Tensor:
        """
        Get one-hot encoded agent IDs for symmetry breaking.

        Returns:
            Tensor of shape (n_agents, n_agents) with one-hot encoding
        """
        return torch.eye(self.n_agents, device=self.device)

    def _add_messages_and_ids_to_obs(self, base_obs: torch.Tensor) -> torch.Tensor:
        """
        Add received messages and agent IDs to observations.

        Args:
            base_obs: Base observation tensor from parent class (n_agents, 14)

        Returns:
            Extended observation with messages and IDs (n_agents, 14 + message_dim + id_dim)
        """
        n_agents = base_obs.shape[0]

        if self.comm_range is None:
            # Global broadcast: each agent sees all other agents' messages
            messages = self._get_global_messages(n_agents)
        else:
            # Local broadcast: only nearby agents
            messages = self._get_local_messages(n_agents)

        # Build extended observation
        parts = [base_obs, messages]

        # Add agent IDs if enabled
        if self.include_agent_id:
            agent_ids = self._get_agent_ids()
            parts.append(agent_ids)

        return torch.cat(parts, dim=-1)

    def _get_global_messages(self, n_agents: int) -> torch.Tensor:
        """
        Get one-hot encoded messages from all other agents (global broadcast).

        Returns:
            Tensor of shape (n_agents, (n_agents-1) * vocab_size)
        """
        # One-hot encode all messages
        messages_onehot = torch.nn.functional.one_hot(
            self.message_buffer, num_classes=self.vocab_size
        ).float()  # (n_agents, vocab_size)

        # For each agent, gather messages from all OTHER agents
        all_messages = []
        for i in range(n_agents):
            # Get all messages except agent i's own message
            mask = torch.ones(n_agents, dtype=torch.bool, device=self.device)
            mask[i] = False
            other_messages = messages_onehot[mask].flatten()  # (n_agents-1) * vocab_size
            all_messages.append(other_messages)

        return torch.stack(all_messages)  # (n_agents, (n_agents-1) * vocab_size)

    def _get_local_messages(self, n_agents: int) -> torch.Tensor:
        """
        Get one-hot encoded messages from nearby agents only (local broadcast).

        Returns:
            Tensor of shape (n_agents, (n_agents-1) * vocab_size)
            Zero-padded for agents beyond comm_range
        """
        positions = self.agent_positions
        distances = torch.cdist(positions, positions)  # (n_agents, n_agents)

        # One-hot encode all messages
        messages_onehot = torch.nn.functional.one_hot(
            self.message_buffer, num_classes=self.vocab_size
        ).float()  # (n_agents, vocab_size)

        # For each agent, gather messages from nearby agents
        all_messages = []
        for i in range(n_agents):
            # Find agents within comm_range (excluding self)
            in_range = (distances[i] <= self.comm_range) & (distances[i] > 0)

            # Create message tensor with fixed size (n_agents-1) * vocab_size
            # Padding with zeros for agents out of range
            agent_messages = torch.zeros((n_agents - 1) * self.vocab_size, device=self.device)

            # Get indices of agents in range
            in_range_indices = in_range.nonzero(as_tuple=True)[0]

            # Place messages from nearby agents
            for idx, agent_idx in enumerate(in_range_indices[:n_agents-1]):  # Cap at n_agents-1
                msg_one_hot = messages_onehot[agent_idx]
                agent_messages[idx * self.vocab_size:(idx + 1) * self.vocab_size] = msg_one_hot

            all_messages.append(agent_messages)

        return torch.stack(all_messages)  # (n_agents, (n_agents-1) * vocab_size)

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset environment and initialize message buffer and room tracking."""
        # Call parent reset
        td = super()._reset(tensordict)

        # Initialize message buffer with zeros (message 0 = "Follow me")
        self.message_buffer = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

        # Reset room tracking (no locking - agents move freely)
        self.agent_in_room = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.agent_excluded = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.excluded_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.room_occupancy = torch.zeros(self.n_rooms, dtype=torch.long, device=self.device)
        self.prev_agent_in_room = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)

        # === NEW: Reset advanced tracking ===
        self.prev_agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.room_persistence_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        self.room_entry_order = {i: [] for i in range(self.n_rooms)}
        self.global_step = 0
        self.received_room_full_warning = {i: set() for i in range(self.n_agents)}

        # Add messages and IDs to observations
        base_obs = td["observation"]
        extended_obs = self._add_messages_and_ids_to_obs(base_obs)
        td["observation"] = extended_obs

        return td

    def _compute_room_assignments(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        UNIFIED room assignment with capacity-correct winner selection.

        This is the SINGLE AUTHORITATIVE function for room assignments.
        All other code (rewards, metrics, observations) should use this.

        Algorithm:
        1. Each agent "attempts" only its NEAREST room among rooms it is inside
           (handles edge case of overlapping rooms)
        2. For each room, sort attempting agents by distance
        3. Accept closest `capacity` agents as WINNERS
        4. Mark remaining agents as EXCLUDED (attempted but lost capacity)
        5. Track which room excluded agents lost (for cooldown/re-targeting)

        Returns:
            assignments: (n_agents,) room index or -1 if not assigned
            in_room: (n_agents,) bool, True if successfully in a room (winner)
            excluded: (n_agents,) bool, True if attempted room but lost capacity
            excluded_room_idx: (n_agents,) room they lost or -1
            room_occupancy: (n_rooms,) count of winners per room
        """
        room_dists = torch.cdist(self.agent_positions, self.room_positions)

        # Initialize outputs
        assignments = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        excluded = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        excluded_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        # Step 1: Each agent attempts only its NEAREST room among rooms it's inside
        # This prevents double-assignment when rooms overlap
        inside_any = room_dists < self.room_radius  # [A, R]
        attempt = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        for i in range(self.n_agents):
            inside_rooms = inside_any[i].nonzero(as_tuple=True)[0]
            if inside_rooms.numel() > 0:
                # Pick the closest room among those the agent is inside
                attempt[i] = inside_rooms[room_dists[i, inside_rooms].argmin()]

        # Step 2: For each room, assign winners among attempting agents
        for r in range(self.n_rooms):
            # Only consider agents who are ATTEMPTING this specific room
            attempting_mask = attempt == r
            attempting_indices = attempting_mask.nonzero(as_tuple=True)[0]

            if attempting_indices.numel() == 0:
                continue

            # Sort by distance to this room (closest first)
            distances = room_dists[attempting_indices, r]
            sorted_order = distances.argsort()
            sorted_agents = attempting_indices[sorted_order]

            # Winners: closest `capacity` agents
            winners = sorted_agents[:self.room_capacity]
            assignments[winners] = r

            # Losers: agents beyond capacity (excluded from THIS room)
            if sorted_agents.numel() > self.room_capacity:
                losers = sorted_agents[self.room_capacity:]
                excluded[losers] = True
                excluded_room_idx[losers] = r  # Track which room they lost

        # Compute derived values
        in_room = assignments >= 0
        room_occupancy = torch.zeros(self.n_rooms, dtype=torch.long, device=self.device)
        for r in range(self.n_rooms):
            room_occupancy[r] = (assignments == r).sum()

        return assignments, in_room, excluded, excluded_room_idx, room_occupancy

    def _update_room_tracking(self):
        """
        Update which room each agent is currently in.

        CHANGED: Room tracking now happens in BOTH phases so rewards work correctly.
        Observations are still phase-gated (room info hidden during spinning),
        but the actual room state is tracked for reward computation.

        Uses unified _compute_room_assignments() for consistency.
        """
        # BOTH PHASES: Compute room assignments so rewards work correctly
        # Save previous room assignments for persistence tracking
        self.prev_agent_room_idx = self.agent_room_idx.clone()
        self.prev_agent_in_room = self.agent_in_room.clone()

        # Get authoritative assignments
        assignments, in_room, excluded, excluded_room_idx, room_occupancy = self._compute_room_assignments()

        self.agent_room_idx = assignments
        self.agent_in_room = in_room
        self.agent_excluded = excluded
        self.excluded_room_idx = excluded_room_idx  # Which room excluded agents lost
        self.room_occupancy = room_occupancy

        # Update persistence and entry order tracking
        for i in range(self.n_agents):
            current_room = assignments[i].item()
            prev_room = self.prev_agent_room_idx[i].item()

            if current_room >= 0:
                # Agent is in a room (winner)
                if prev_room != current_room:
                    # New entry to this room
                    if i not in self.room_entry_order[current_room]:
                        self.room_entry_order[current_room].append(i)
                    # Reset persistence since changed rooms
                    self.room_persistence_count[i] = 1
                    # If agent left a room, remove from that room's entry order
                    if prev_room >= 0 and i in self.room_entry_order[prev_room]:
                        self.room_entry_order[prev_room].remove(i)
                else:
                    # Stayed in same room - increment persistence
                    self.room_persistence_count[i] += 1
            else:
                # Agent not in any room (or excluded)
                if prev_room >= 0 and i in self.room_entry_order[prev_room]:
                    self.room_entry_order[prev_room].remove(i)
                # Reset persistence
                self.room_persistence_count[i] = 0

    def _update_communication_warnings(self):
        """
        Track which agents received 'Room Full' (M1) warnings from agents in rooms.

        This allows penalizing agents who enter rooms despite warnings.
        """
        # Clear previous warnings
        for i in range(self.n_agents):
            self.received_room_full_warning[i] = set()

        # Check each agent's messages
        for sender_idx in range(self.n_agents):
            msg = self.message_buffer[sender_idx].item()

            # If sender is sending "Room Full" (M1) and is in a room
            if msg == MSG_ROOM_FULL and self.agent_in_room[sender_idx]:
                sender_room = self.agent_room_idx[sender_idx].item()

                # All other agents receive this warning about sender's room
                for receiver_idx in range(self.n_agents):
                    if receiver_idx != sender_idx:
                        self.received_room_full_warning[receiver_idx].add(sender_room)

    def get_agent_entry_rank(self, agent_idx: int, room_idx: int) -> int:
        """
        Get the entry rank of an agent in a room.

        Returns:
            Entry rank (0 = first, 1 = second, etc.) or -1 if not in room
        """
        if room_idx < 0 or room_idx >= self.n_rooms:
            return -1
        entry_list = self.room_entry_order[room_idx]
        if agent_idx in entry_list:
            return entry_list.index(agent_idx)
        return -1

    def agent_ignored_warning(self, agent_idx: int) -> bool:
        """
        Check if agent entered a room despite receiving a 'Room Full' warning.

        Returns:
            True if agent is in a room they were warned about
        """
        if not self.agent_in_room[agent_idx]:
            return False
        current_room = self.agent_room_idx[agent_idx].item()
        return current_room in self.received_room_full_warning[agent_idx]

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """
        Step environment with actions and messages.

        Expected tensordict keys:
            - "action": (n_agents, 2) movement actions
            - "message": (n_agents,) discrete message indices (optional - will be auto-set based on state)

        Agents move freely - no freezing. Room tracking is updated for rewards.

        MESSAGE LOGIC (state-based):
        - Message is automatically determined by room state, NOT by policy output
        - If agent is in a FULL room (occupancy >= capacity) → "Room full" (1)
        - Otherwise → "Follow me" (0)
        """
        # Agents move freely - no freezing
        td = super()._step(tensordict)

        # Update room tracking after movement (for reward computation)
        self._update_room_tracking()

        # AUTO-SET MESSAGES BASED ON STATE (not policy output)
        # This ensures messages have MEANING based on actual room state
        self._update_state_based_messages()

        # Update communication warning tracking
        self._update_communication_warnings()

        # Increment global step counter
        self.global_step += 1

        # Add messages and IDs to next observation
        base_obs = td["observation"]
        extended_obs = self._add_messages_and_ids_to_obs(base_obs)
        td["observation"] = extended_obs

        return td

    def _update_state_based_messages(self):
        """
        Automatically set messages based on phase and room state.

        SPINNING PHASE:
        - 1-2 random agents say "Follow me" (leaders)
        - Others stay silent (no message / default)

        CLAIMING PHASE:
        - If agent is in a room that is FULL (occupancy >= capacity) -> "Room full" (1)
        - Otherwise -> "Follow me" (0)

        This ensures messages reflect actual state and enable coordination.
        """
        # Default: everyone says "Follow me"
        self.message_buffer = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)

        if self.phase == "spinning":
            # SPINNING PHASE: 1-2 random agents are "leaders" saying "Follow me"
            # Initialize leaders once at start of spinning phase
            if not hasattr(self, '_spinning_leaders') or self._spinning_leaders is None:
                # Randomly select 1-2 leaders
                num_leaders = torch.randint(1, 3, (1,)).item()  # 1 or 2 leaders
                leader_indices = torch.randperm(self.n_agents)[:num_leaders]
                self._spinning_leaders = set(leader_indices.tolist())

            # Only leaders say "Follow me" (0), others say nothing (we use -1 to indicate no message)
            # But since vocab only has 0 and 1, we keep all as 0 but track leaders separately
            # Leaders are shown with "Follow me", non-leaders shown without message indicator

        else:
            # CLAIMING PHASE: state-based messages
            # Reset spinning leaders when entering claiming phase
            self._spinning_leaders = None

            # Check each agent's room status
            for i in range(self.n_agents):
                if self.agent_in_room[i]:
                    # Agent is in a room - check if room is full
                    room_idx = self.agent_room_idx[i].item()
                    if room_idx >= 0:
                        room_count = self.room_occupancy[room_idx].item()
                        if room_count >= self.room_capacity:
                            # Room is full - set message to "Room full"
                            self.message_buffer[i] = MSG_ROOM_FULL

    def reconfigure(self, n_agents: int = None, n_rooms: int = None, room_capacity: int = None):
        """
        Reconfigure environment for curriculum learning.

        This allows changing the number of agents/rooms without recreating the environment.

        Args:
            n_agents: New number of agents (or None to keep current)
            n_rooms: New number of rooms (or None to keep current)
            room_capacity: New room capacity (or None to keep current)
        """
        if n_agents is not None:
            self.n_agents = n_agents
        if n_rooms is not None:
            self.n_rooms = n_rooms
        if room_capacity is not None:
            self.room_capacity = room_capacity

        # Recalculate observation dimension
        messages_obs_dim = (self.n_agents - 1) * self.vocab_size
        agent_id_dim = self.n_agents if self.include_agent_id else 0
        new_obs_dim = 14 + messages_obs_dim + agent_id_dim

        self._messages_obs_dim = messages_obs_dim
        self._agent_id_dim = agent_id_dim

        # Update specs
        self.observation_spec = Composite({
            "observation": Unbounded(shape=(self.n_agents, new_obs_dim), device=self.device),
        })

        self.action_spec = Composite({
            "action": Bounded(
                low=-self.max_speed,
                high=self.max_speed,
                shape=(self.n_agents, 2),
                device=self.device,
            ),
            "message": Categorical(
                n=self.vocab_size,
                shape=(self.n_agents,),
                device=self.device,
            ),
        })

        # Reinitialize room positions
        self._init_rooms()

        # Reset message buffer and room tracking for new agent count
        self.message_buffer = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        self.agent_in_room = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        self.agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)

        # Reset advanced tracking for fair rewards
        self.prev_agent_room_idx = torch.full((self.n_agents,), -1, dtype=torch.long, device=self.device)
        self.room_persistence_count = torch.zeros(self.n_agents, dtype=torch.long, device=self.device)
        self.room_entry_order = {i: [] for i in range(self.n_rooms)}
        self.global_step = 0
        self.received_room_full_warning = {i: set() for i in range(self.n_agents)}

    def get_message_meanings(self):
        """Return message meanings for analysis."""
        return {
            MSG_FOLLOW_ME: "Follow me",
            MSG_ROOM_FULL: "Room is full"
        }
