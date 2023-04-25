import numpy as np
import pandas as pd
import networkx as nx
import copy, gym, igraph

import jax
from jax import numpy as jnp

from scipy.special import softmax
from hypotheses import FULL, HYPS


# Full set of interventions
INTERV = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ],
    dtype=jnp.float32,
)


class Policy:
    """Abstract class for policy"""

    def __init__(self) -> None:
        pass

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): agent state

        Returns:
            np.ndarray: action
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs) -> None:
        """Reset policy"""
        pass


class RandomPolicy(Policy):
    """Random samples from action space

    Args:
        action_space (gym.Space): action space

    Attributes:
        action_space (gym.Space): action space
    """

    def __init__(self, *, action_space: gym.Space) -> None:
        self.action_space = action_space

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        return self.action_space.sample()


class FixedInterventionPolicy(Policy):
    """Policy that performs fixed interventions on graph (actions)

    Args:
        action_space (gym.Space): action space

    Attributes:
        action_space (gym.Space): action space
    """

    def __init__(self, *, action_space: gym.Space, value=None) -> None:
        self.action_space = action_space
        self.value = value if value is not None else action_space.low

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        return self.value


class CausalPolicy(Policy):
    """Simple structure discovery with score-based methods, random interventions

    Args:
        obs_shape (int): observation shape
        action_shape (int): action shape
    """

    def __init__(
        self, *, obs_shape: int, action_shape: int, graph: np.ndarray, model
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.graph = graph
        self.model = model

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        self.obs_buffer = np.append(
            self.obs_buffer, state[None, : self.obs_shape], axis=0
        )
        data = pd.DataFrame(copy.deepcopy(self.obs_buffer))

        if state.shape[-1] > self.obs_shape and self.step_cnt > 0:
            output_graph = self.model.create_graph_from_data(data)

            self.step_cnt += 1
            return (
                self.action_space.sample(),
                nx.adjacency_matrix(output_graph).todense(),
            )
        else:
            self.step_cnt += 1
            return self.action_space.sample(), np.ones((4, 4)).astype(int)

    def reset(self) -> None:
        """Reset policy"""
        self.obs_buffer = np.empty((0, self.obs_shape))
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        self.step_cnt = 0


class EGreedyLLPolicy:
    """Simple math:`\\epsilon`-greedy policy based on maximum log-likelihood math:`\\log p(D | G, \\Theta)`.

    Args:
        inference_model (dibs.inference.Dibs): inference model
        theta (np.ndarray): sampled parameters
        gs (np.ndarray): sampled graphs
        inter_mask (np.ndarray): intervention masks
        action_shape (int): action shape
        n_step (int): number of steps to look behind
        force_exploration (int): number of steps to force exploration,
        buffer (class): buffer to store intervention data
        best_particles (dict): dictionary of best particles for each hypotheses
        topk (int): number of best particles to consider wrt. log-likelihood
    """

    def __init__(
        self,
        *,
        inference_model,
        thetas,
        gs,
        inter_mask,
        action_space: int,
        n_step: int,
        force_exploration: int,
        buffer,
        best_particles: dict,
        topk: int,
    ) -> None:
        self.inference_model = inference_model
        self.thetas = thetas
        self.gs = gs
        self.inter_mask = jnp.array(inter_mask)
        self.action_space = action_space
        self.n_step = n_step
        self.force_exploration = force_exploration
        self.buffer = buffer
        self.best_particles = best_particles
        self.K = topk

        self.log_d_g_t = jax.vmap(
            lambda x, single_g, single_theta: self.inference_model.log_likelihood(
                x=x,
                theta=single_theta,
                g=single_g,
                interv_targets=self.inter_mask[: x.shape[0], ...],
            ),
            (None, 0, 0),
            0,
        )

        self.log_d_g_t_score = jax.vmap(
            lambda x, d, single_g, single_theta: self.inference_model.log_likelihood(
                x=jnp.concatenate([d, x], axis=0),
                theta=single_theta,
                g=single_g,
                interv_targets=self.inter_mask[: d.shape[0] + 1, ...],
            ),
            (0, None, 0, 0),
            0,
        )

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        # Perform random intervention if we are in the first few steps
        if self.force_exploration > self.step_cnt:
            self.step_cnt += 1
            return self.action_space.sample(), FULL

        else:
            # Update buffer and compute log likelihood of data given sampled parameters
            self.buffer(state=state)

            # Compute the log likelihood of the data given the sampled parameters
            log_d_g_t_list = self.log_d_g_t(
                jnp.array(self.buffer.data), self.gs, self.thetas
            )
            log_d_g_t_list = np.array(log_d_g_t_list)

            # Pick the topk particle that maximizes the log likelihood
            mll = np.argpartition(log_d_g_t_list, -self.K)[-self.K :][::-1]

            # Belief set over hypotheses
            belief_set = set()
            for key in self.best_particles:
                for ml in mll:
                    if ml in self.best_particles[key]:
                        belief_set.add(key)
            belief_set = list(belief_set)
            # print(f"Belief set: {belief_set}")

            self.step_cnt += 1
            return self.construct_action(mll, log_d_g_t_list[mll])

    def construct_action(self, mll, log_d_g_t_belief_set):
        """Construct action given belief set and particle index

        Args:
            mll (np.ndarray): index of particles that maximize the log likelihood
            log_d_g_t_belief_set: Log likelihood of data given the belief set
            best_particles (dict): dictionary of best particles for each hypotheses

        Returns:
            action: intervention action
        """
        # pick a hypothesis from the belief set
        hyp = np.random.choice(mll, p=softmax(np.exp(log_d_g_t_belief_set)))

        z_idx = np.random.choice(["A", "B", "C", "AB", "AC", "BC", "ABC"])
        for key in self.best_particles.keys():
            if hyp in self.best_particles[key]:
                z_idx = key
                break

        action = np.random.randint(2, size=len(self.action_space))
        val = np.random.randint(0, 2)
        for idx, symbol in enumerate(["A", "B", "C"]):
            if symbol in z_idx:
                action[idx] = val
        return action, HYPS[z_idx]

    def reset(self, *args, **kwargs) -> None:
        """Reset policy"""
        self.buffer.reset()
        self.step_cnt = 0


class RandomLLPolicy:
    """Simple random policy based on maximum log-likelihood math:`\\log p(D | G, \\Theta)`.

    Args:
        inference_model (dibs.inference.Dibs): inference model
        theta (np.ndarray): sampled parameters
        gs (np.ndarray): sampled graphs
        inter_mask (np.ndarray): intervention masks
        action_shape (int): action shape
        n_step (int): number of steps to look behind
        force_exploration (int): number of steps to force exploration,
        buffer (class): buffer to store intervention data
        best_particles (dict): dictionary of best particles for each hypotheses
    """

    def __init__(
        self,
        *,
        inference_model,
        thetas,
        gs,
        inter_mask,
        action_space: int,
        n_step: int,
        force_exploration: int,
        buffer,
        best_particles: dict,
    ) -> None:
        self.inference_model = inference_model
        self.thetas = thetas
        self.gs = gs
        self.inter_mask = jnp.array(inter_mask)
        self.action_space = action_space
        self.n_step = n_step
        self.force_exploration = force_exploration
        self.buffer = buffer
        self.best_particles = best_particles

        self.log_d_g_t = jax.vmap(
            lambda x, single_g, single_theta: self.inference_model.log_likelihood(
                x=x,
                theta=single_theta,
                g=single_g,
                interv_targets=self.inter_mask[: x.shape[0], ...],
            ),
            (None, 0, 0),
            0,
        )

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        # Perform random intervention if we are in the first few steps
        if self.force_exploration > self.step_cnt:
            self.step_cnt += 1
            return self.action_space.sample(), FULL

        else:
            # Update buffer and compute log likelihood of data given sampled parameters
            self.buffer(state=state)

            # Compute the log likelihood of the data given the sampled parameters
            log_d_g_t_list = self.log_d_g_t(
                jnp.array(self.buffer.data), self.gs, self.thetas
            )
            log_d_g_t_list = np.array(log_d_g_t_list)

            # Pick the topk particle that maximizes the log likelihood
            mll = np.argmax(log_d_g_t_list)

            # Belief set over hypotheses
            belief_set = set()
            for key in self.best_particles:
                if mll in self.best_particles[key]:
                    belief_set.add(key)
            belief_set = list(belief_set)
            if belief_set == []:
                belief_set = ['FULL']
            # print(f"Belief set: {belief_set}")

            self.step_cnt += 1
            return self.action_space.sample(), HYPS[np.random.choice(belief_set)]

    def reset(self, *args, **kwargs) -> None:
        """Reset policy"""
        self.buffer.reset()
        self.step_cnt = 0


class BALDPolicy:
    """Simple math:`\\epsilon`-greedy policy based on maximum log-likelihood math:`\\log p(D | G, \\Theta)`.

    Args:
        inference_model (dibs.inference.Dibs): inference model
        theta (np.ndarray): sampled parameters
        gs (np.ndarray): sampled graphs
        inter_mask (np.ndarray): intervention masks
        action_shape (int): action shape
        n_step (int): number of steps to look behind
        force_exploration (int): number of steps to force exploration,
        buffer (class): buffer to store intervention data
        best_particles (dict): dictionary of best particles for each hypotheses
        topk (int): number of best particles to consider wrt. log-likelihood
    """

    def __init__(
        self,
        *,
        inference_model,
        theta,
        thetas,
        gs,
        inter_mask,
        action_space: int,
        n_step: int,
        force_exploration: int,
        buffer,
        best_particles: dict,
        topk: int,
        key
    ) -> None:
        self.inference_model = inference_model
        self.theta = theta
        self.thetas = thetas
        self.gs = gs
        self.inter_mask = jnp.array(inter_mask)
        self.action_space = action_space
        self.n_step = n_step
        self.force_exploration = force_exploration
        self.buffer = buffer
        self.best_particles = best_particles
        self.K = topk
        self.key = key

        self.log_d_g_t = jax.vmap(
            lambda x, single_g, single_theta: self.inference_model.log_likelihood(
                x=x,
                theta=single_theta,
                g=single_g,
                interv_targets=self.inter_mask[: x.shape[0], ...],
            ),
            (None, 0, 0),
            0,
        )

        self.rollout_exp_obs = jax.vmap(
            lambda key, single_g, single_theta, interv_val: self.inference_model.sample_obs(
                key=key,
                n_samples=1,
                g=igraph.Graph.Adjacency(single_g.tolist()),
                theta=single_theta,
                toporder=None,
                interv={i: v for i, v in zip(range(single_g.shape[0]), interv_val)},
            ),
            (None, None, None, 0),
            0,
        )

    def __call__(self, *, state: np.ndarray) -> np.ndarray:
        """Perform action given state

        Args:
            state (np.ndarray): state of agent

        Returns:
            np.ndarray: action
        """
        # Perform random intervention if we are in the first few steps
        if self.force_exploration > self.step_cnt:
            self.step_cnt += 1
            return self.action_space.sample(), FULL

        else:
            # Update buffer and compute log likelihood of data given sampled parameters
            self.buffer(state=state)

            # Compute the log likelihood of the data given the sampled parameters
            log_d_g_t_list = self.log_d_g_t(
                jnp.array(self.buffer.data), self.gs, self.theta
            )
            log_d_g_t_list = np.array(log_d_g_t_list)

            # Pick the topk particle that maximizes the log likelihood
            mll = np.argpartition(log_d_g_t_list, -self.K)[-self.K :][::-1]

            output = np.empty((0, 8))
            for z_idx in mll:
                self.key, subk = jax.random.split(self.key)
                out = self.rollout_exp_obs(subk, self.gs[z_idx], self.thetas[z_idx], INTERV)
                output = np.vstack((output, np.where(out[None, :, 0, -1] > 0.5, 1, 0)))
            inter = INTERV[np.argmax(output.var(axis=0, ddof=1)), ...]

            # pick a hypothesis from the belief set
            hyp = np.random.choice(mll, p=softmax(np.exp(log_d_g_t_list[mll])))

            z_idx = np.random.choice(["A", "B", "C", "AB", "AC", "BC", "ABC"])
            for key in self.best_particles.keys():
                if hyp in self.best_particles[key]:
                    z_idx = key
                    break

            self.step_cnt += 1
            return inter, HYPS[z_idx]

    def reset(self, *args, **kwargs) -> None:
        """Reset policy"""
        self.buffer.reset()
        self.step_cnt = 0