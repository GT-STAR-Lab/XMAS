from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.transforms import _apply_to_composite
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
import numpy as np
import torch

class ConnectivityGraph(Transform):
    """
    Tracks the communication graph at each time step
    of the environment.

    Generates a tensor of shape (num_environments, num_agents, num_agents, num_agents)
    TODO: Is this the correct way to generate the communication graph?
    """
    def __init__(
            self,
            in_keys=None,
            out_keys=None,
            reset_keys=None,
            conn_type='full',
            conn_radius=None,
            n_agents=None,
            pos_indicies=None
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys
        self.conn_type = conn_type
        self.conn_radius = conn_radius
        self.n_agents = n_agents
        self.default_edge_index = torch.Tensor(1 - np.eye(self.n_agents, self.n_agents))

    def _apply_transform(self, obs: torch.Tensor)->None:
        """
        Applies the transform to a tensor.
        """
        if self.conn_type == 'full':
            edge_index = torch.Tensor(1 - np.eye(self.n_agents, self.n_agents))
            edge_index = edge_index.repeat(*obs.shape[:2], 1, 1)
        elif self.conn_type == 'radius':
            # Add edges between agents that are within range of eachother
            agent_positions = obs[:, :, :2].unsqueeze(2)
            dists = torch.norm(agent_positions - agent_positions.transpose(1,2), dim=3)
            edge_index = ((dists > 0) & (dists <= self.conn_radius)).float()

            # Duplicate the same graph for each agent
            edge_index = edge_index[:, None, :, :].repeat(1, self.n_agents, 1, 1)

        return(edge_index)

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``  
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        """
        Transform the observation spec to match the ConnectivityTransform mapping.
        """
        return UnboundedContinuousTensorSpec(
            shape=(*observation_spec.shape[:2], self.n_agents, self.n_agents),
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )


class RemoveLidar(Transform):
    """
    Remove LIDAR from agent observations.
    """
    def __init__(
            self,
            in_keys=None,
            out_keys=None,
            reset_keys=None,
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys

    def _apply_transform(self, obs: torch.Tensor)->None:
        """
        Applies the transform to a tensor.
        """

        # slice the last 12 elements off of the observation, corresponding to LIDAR
        return obs[:, :, :-12]

    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``  
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        """
        Transform the observation spec to match 
        """
        return UnboundedContinuousTensorSpec(
            shape=(*observation_spec.shape[:2], observation_spec.shape[2]-12),
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )

class AddAttn(Transform):
    """
    Add attn.
    """
    def __init__(
            self,
            in_keys=None,
            out_keys=None,
            reset_keys=None,
            n_agents=3
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys
        self.n_agents = n_agents

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        attn_probs = tensordict.get(("agents", "attn"), None)
        if attn_probs is None:
            self.n_agents = tensordict['agents'].batch_size[1]
            # attn_probs = torch.tensor([1/self.n_agents]).repeat(tensordict['agents'].batch_size[0],self.n_agents,self.n_agents)
            attn_probs = torch.tensor([1/self.n_agents]).repeat(tensordict['agents'].batch_size[0],self.n_agents,self.n_agents).to(tensordict.device)
            tensordict['agents', 'attn'] = attn_probs
        agent_obs = tensordict['agents', 'observation']
        tensordict.set(("agents", "observation"), torch.cat((agent_obs, attn_probs), -1))
        
        return tensordict
    
    # _apply_to_composite will execute the observation spec transform across all
    # in_keys/out_keys pairs and write the result in the observation_spec which
    # is of type ``Composite``  
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        """
        Transform the observation spec to match 
        """
        return UnboundedContinuousTensorSpec(
            shape=(*observation_spec.shape[:2], observation_spec.shape[2]+self.n_agents),
            dtype=observation_spec.dtype,
            device=observation_spec.device,
        )