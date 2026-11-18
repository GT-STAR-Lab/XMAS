import torch
import numpy as np

from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torchrl.data import DEVICE_TYPING
from typing import Optional
from .gnn import GCNConv, GATV2Conv
from torch_geometric.nn import MessagePassing

from torchrl.modules import ProbabilisticActor, TanhNormal, MultiAgentMLP, MLP

class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        c_hidden=32,
        num_layers=1,
        attention=False,
        n_environments=60,
        n_agents=3,
        dp_rate=0.1,
        passage=False,
        discovery=False,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = GATV2Conv if attention else GCNConv

        self.n_environments = n_environments
        self.n_agents = n_agents
        self.c_in = c_in
        self.c_out = c_out
        self.explain_yourself = False
        self.return_alpha = kwargs.get('return_alpha', False)
        self.dp_rate = dp_rate
        if dp_rate != 0:
            print('using dp: ', dp_rate)
        
        # passsage = False
        layers = []
        if passage:
            layers = [MultiAgentMLP(n_agent_inputs=c_in,n_agent_outputs=c_in,centralised=False,share_params=True,depth=4,num_cells=32,activation_class=torch.nn.ReLU,n_agents=self.n_agents, device=kwargs['device'])]
        elif discovery:
            layers = [MultiAgentMLP(n_agent_inputs=c_in,n_agent_outputs=c_in,centralised=False,share_params=True,depth=1,num_cells=64,activation_class=torch.nn.ReLU,n_agents=self.n_agents, device=kwargs['device'])]
            c_hidden = 64
        # layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            if attention:
                layers += [
                    gnn_layer(in_channels=in_channels, out_channels=out_channels, share_weights=kwargs['share_weights'], return_alpha=kwargs['return_alpha'], device=kwargs['device'], dp_rate=dp_rate),
                    nn.Tanh(),
                ]
            else:
                layers += [
                    gnn_layer(in_channels=in_channels, out_channels=out_channels, device=kwargs['device']),
                    nn.Tanh(),
                ]
            in_channels = c_hidden + c_in
        # layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        in_channels = c_hidden + c_in
        self.in_channels = in_channels
        if passage:
            layers += [MultiAgentMLP(n_agent_inputs=in_channels,n_agent_outputs=c_out,centralised=False,share_params=True,depth=4,num_cells=64,activation_class=torch.nn.ReLU,n_agents=self.n_agents, device=kwargs['device'])]
        elif discovery:
            layers += [MultiAgentMLP(n_agent_inputs=in_channels,n_agent_outputs=c_out,centralised=False,share_params=True,depth=2,num_cells=64,activation_class=torch.nn.ReLU,n_agents=self.n_agents, device=kwargs['device'])]
        else:
            layers += [MultiAgentMLP(n_agent_inputs=in_channels,n_agent_outputs=c_out,centralised=False,share_params=True,depth=2,num_cells=256,activation_class=torch.nn.Tanh,n_agents=self.n_agents, device=kwargs['device'])]
        # layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)
        print(self.layers)
        self.num_layers = num_layers

    def forward(self, x, edge_index, **kwargs):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        input = x
        prev_alpha = None
        num_attn_layers = 0
        for i, layer in enumerate(self.layers):
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, MessagePassing):
                if isinstance(layer, GCNConv) or (isinstance(layer, GATV2Conv) and self.return_alpha == False):
                    x = layer(x, edge_index, self.dp_rate)
                elif isinstance(layer, GATV2Conv) and self.return_alpha == True:
                    num_attn_layers = num_attn_layers + 1
                    x, alpha = layer(x, edge_index, self.dp_rate)
                    if prev_alpha is None:
                        prev_alpha = alpha
                    else:
                        prev_alpha = prev_alpha + alpha
                if i < self.num_layers:
                    x = torch.cat((x, input), dim=1)
                else:
                    self.n_environments = x.shape[0]//(self.n_agents)
                    x = x.view(self.n_environments, self.n_agents, self.c_out)
            elif isinstance(layer, MultiAgentMLP) and i > 0:
                # x = torch.cat((x, input), dim=1)
                self.n_environments = x.shape[0]//(self.n_agents)
                x = x.view(self.n_environments, self.n_agents, self.in_channels)
                x = layer(x)
            elif isinstance(layer, MultiAgentMLP) and i == 0:
                if i == 0:
                    self.n_environments = x.shape[0]//(self.n_agents)
                    x = x.view(self.n_environments, self.n_agents, self.c_in)
                x = layer(x)
                x = x.view(self.n_environments*self.n_agents, self.c_in)
                # print('tanh')
            else:
                x = layer(x)
        if kwargs.get('unfaithful_save_og', False):
            self.og_output = x
        if self.return_alpha == True:
            return x, prev_alpha/num_attn_layers
        return x if self.explain_yourself == False else x.chunk(2, -1)[0]


class MultiAgentGNN(nn.Module):
    """
    Multi-Agent GNN

    Attributes:
        n_agents (int): number of agents.
        n_agent_inputs (int): size of agent's observation vector.
        gnn_agent_outputs (int): size of agent's node embedding obtained from gnn.
        n_agent_outputs (int): size of agent's outputs.
        attention (bool): if true, use GAT layer, otherwise GCN.
        return_alpha (bool): if true and attention is true, return attn matrix.
        agent_networks (nn.ModuleList): ModuleList of gnn networks.
        share_params (bool): if true, use same gnn network for all agents.
        device (DEVICE_TYPING): cpu or cuda.
    """
    def __init__(
        self,
        n_agent_inputs: int,
        gnn_agent_outputs: int,
        n_agent_outputs: int,
        n_agents: int,
        attention: bool,
        share_params: bool,
        device: Optional[DEVICE_TYPING] = None,
        action_output=False,
        num_layers=1,
        dp_rate=0.1,
        passage=False,
        discovery=False,
        **kwargs
    ):
        """
        Initialize a MultiAgentGNN.

        Args:
            n_agent_inputs (int): size of agent's observation vector.
            gnn_agent_outputs (int): size of agent's node embedding obtained from gnn.
            n_agent_outputs (int): size of agent's outputs.
            n_agents (int): number of agents.
            share_params (bool): if true, use same gnn network for all agents.
            attention (bool): if true, use GAT layer, else GCN layer.
            device (DEVICE_TYPING): cpu or cuda

        Raises:
            ValueError: If output per agent is not equal to size of input per agent +
            size of agent's node embedding obtained from gnn
        """
        # if n_agent_outputs != n_agent_inputs + gnn_agent_outputs:
        #     raise ValueError(
        #         f"Multi-agent network expected n_agent_outputs to equal n_agent_inputs + gnn_agent_outputs"
        #         f"{n_agent_inputs + gnn_agent_outputs}, but got {n_agent_outputs}"
        #     )
        
        super().__init__()

        self.device = device

        self.action_output = action_output

        self.num_layers = num_layers

        self.dp_rate = dp_rate

        # number of agents
        self.n_agents = n_agents

        # size of inputs per agent
        self.n_agent_inputs = n_agent_inputs

        # size of outputs per agent from message passing
        self.gnn_agent_outputs = gnn_agent_outputs

        # size of outputs per agent
        self.n_agent_outputs = n_agent_outputs

        # indicate whether a shared graph is used among all agents
        self.share_params = share_params

        # indicate whether to use attention
        self.attention = attention

        # indicate whether to return attention matrix for visualization
        self.return_alpha = kwargs['return_alpha'] and self.attention if 'return_alpha' in kwargs else False

        # list of networks per agent
        if self.num_layers > 1:
            self.agent_networks = nn.ModuleList(
                [
                    GNNModel(
                        c_in=self.n_agent_inputs,
                        c_out=self.gnn_agent_outputs,
                        num_layers=self.num_layers,
                        attention=self.attention,
                        share_weights=True,
                        return_alpha=self.return_alpha,
                        device=device,
                        n_agents=self.n_agents,
                        dp_rate=self.dp_rate,
                        passage=passage,
                        discovery=discovery
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]) if self.attention else nn.ModuleList(
                [
                    GNNModel(
                        c_in=self.n_agent_inputs,
                        c_out=self.gnn_agent_outputs,
                        num_layers=self.num_layers,
                        device=device,
                        n_agents=self.n_agents,
                        dp_rate=self.dp_rate,
                        passage=passage,
                        discovery=discovery
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        else:
            self.agent_networks = nn.ModuleList(
                [
                    GATV2Conv(
                        in_channels=self.n_agent_inputs,
                        out_channels=self.gnn_agent_outputs,
                        share_weights=True,
                        return_alpha=self.return_alpha,
                        device=device,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]) if self.attention else nn.ModuleList(
                [
                    GCNConv(
                        in_channels=self.n_agent_inputs,
                        out_channels=self.gnn_agent_outputs,
                        device=device,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Conduct a round of message passing.

        Args:
            input (torch.Tensor): agent inputs [n_environments, n_agents, n_agent_inputs]
            edge_index (torch.Tensor): adjacency matrix [n_environments, n_agents, n_agents, n_agents]

        Returns:
            (torch.Tensor) network outputs [n_environments, n_agents, n_agent_outputs]

        Raises:
            ValueError: If last two dimensions of input shape are not n_agents, n_agent_inputs
            ValueError: If last two dimensions of output shape are not n_agents, n_agent_outputs
        """
        input = x
        if input.shape[-2:] != (self.n_agents, self.n_agent_inputs):
            raise ValueError(
                f"Multi-agent network expected input with last 2 dimensions {[self.n_agents, self.n_agent_inputs]},"
                f" but got {input.shape}"
            )
        if len(input.shape) == 2:
            batch = Batch.from_data_list([Data(x=input, edge_index=edge_index)])
            # batch data into single graph batch for efficiency using DataLoader
            n_environments = 1
            batch_loader = DataLoader(batch, batch_size=n_environments, shuffle=False)
            # net = self.agent_networks[0]
            # for batch in batch_loader:
            #     if self.return_alpha:
            #         output, attn = net(batch.x, batch.edge_index)
            #         attn = attn.view(n_environments, self.n_agents, self.n_agents)
            #     else: 
            #         output = net(batch.x, batch.edge_index)

            #     output = output.view(self.n_agents, self.gnn_agent_outputs)
            # output = torch.cat((input, output), dim=1)
            # return (output, attn) if self.return_alpha else output
        else:
            n_environments = input.shape[0]

            # currently, within the same environment, each agent has the same graph, just using 0th index for now
            # batch = Batch.from_data_list([Data(x=input[i], edge_index=edge_index[i,:,0].to_sparse().indices()) for i in range(n_environments)])
            batch = Batch.from_data_list([Data(x=input[i], edge_index=torch.tensor(np.triu(edge_index[0][0].cpu())).to(self.device).to_sparse().indices()) for i in range(n_environments)])

            # batch data into single graph batch for efficiency using DataLoader
            batch_loader = DataLoader(batch, batch_size=n_environments, shuffle=False)

        if not self.share_params:
            for batch in batch_loader:
                output = torch.stack(
                    [self.agent_networks[i](batch.x, batch.edge_index, **kwargs).view(n_environments, self.n_agents, self.gnn_agent_outputs)[:,i,:]
                        for i in range(self.n_agents)],
                    dim=1
                )

        # agents use the same network for shared parameters
        else:
            net = self.agent_networks[0]
            for batch in batch_loader:
                if self.return_alpha:
                    output, attn = net(batch.x, batch.edge_index, **kwargs)
                    attn = attn.view(n_environments, self.n_agents, self.n_agents)
                else: 
                    output = net(batch.x, batch.edge_index, **kwargs)

                output = output.view(n_environments, self.n_agents, self.gnn_agent_outputs) if net.explain_yourself == False else output.view(n_environments, self.n_agents, self.gnn_agent_outputs//2)

        # concatenate original observations with output
        if len(input.shape) != 2 and self.action_output == False:
            output = torch.cat((input, output), dim=2)

        # if output.shape[-2:] != (self.n_agents, self.gnn_agent_outputs + self.n_agent_inputs):
        #     raise ValueError(
        #         f"Multi-agent network expected output with last 2 dimensions {[self.n_agents, self.n_agent_outputs]},"
        #         f" but got {output.shape}"
        #     )

        return (output, attn) if self.return_alpha else output
