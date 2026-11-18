import torch
from torch.nn import Linear, Parameter, functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, add_remaining_self_loops, to_undirected, degree, softmax, dropout_edge, dropout_node
from torch_geometric.typing import OptTensor
from torchrl.data import DEVICE_TYPING
from typing import Optional

class GCNConv(MessagePassing):
    """
    Graph convolution layer.

    Attributes:
        lin (nn.Linear): learned linear transform for node features to embeddings.
        bias (nn.Parameter): learned bias vector.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            device: Optional[DEVICE_TYPING] = None,
            dp_rate: float = 0
        ):
        """
        Initialize a graph convolution layer.

        Args:
            in_channels (int): size of each input per node
            out_channels (int): size of each output per node
            device (DEVICE_TYPING): cpu or cuda
        """
        # initialize message passing with sum aggergation scheme
        super().__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.dp_rate = dp_rate

        # linear transform for node features in_channels -> out_channels
        self.lin = Linear(in_channels, out_channels, bias=False, device=device)

        # bias vector
        self.bias = Parameter(torch.empty(out_channels, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        """ Reset model parameters """
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, dp_rate: float):
        """
        Conduct a round of message passing.

        Args:
            x (torch.Tensor): [N, in_channels]
            edge_index (torch.Tensor: [2, E]
        
        Returns:
            (torch.Tensor) [N, out_channels]
        """
        self.dp_rate = dp_rate

        # Step 1: Add self-loops to the adjacency matrix
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = to_undirected(edge_index)
        # edge_index, _ = dropout_edge(edge_index, self.dp_rate)
        og_edge_index = edge_index
        if self.dp_rate != 0:
            # edge_index, edge_mask = dropout_edge(edge_index, self.dp_rate)
            edge_index, edge_mask, _ = dropout_node(edge_index, self.dp_rate)
        # else:
        #     print('skipping dropout')

        # Step 2: Linearly transform node feature matrix
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating message
        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias

        return(out)
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        """
        Construct messages to node i for each edge (j, i)

        Args:
            x_j (torch.Tensor): neighboring node features [E, out_channels]
            norm (torch.Tensor): normalization coefficients

        Returns:
            (torch.Tensor) normalized node features
        """
        # x_j has shape [E, out_channels]. "Lifted vector".
        # contains the source node features for each connected edge (i.e. neighbors)

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class GATV2Conv(MessagePassing):
    """
    Graph attention layer with dynamic attention.
    Reference: 
        "How Attentive are Graph Attention Networks" https://arxiv.org/abs/2105.14491

    Attributes:
        lin_s (nn.Linear): learned linear transform for source node features to embeddings
        lin_t (nn.Linear): learned linear transform for target node features to embeddings
        attn (nn.Linear): linear layer to compute attention score
        bias (nn.Parameter): learned bias vector
        alpha (torch.Tensor): store computed attention
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            share_weights: bool,
            return_alpha: bool = False,
            device: Optional[DEVICE_TYPING] = None,
            dp_rate: float = 0
        ):
        """
        Initialize a graph attention layer.

        Args:
            in_channels (int): size of each input per node
            out_channels (int): size of each output per node
            share_weights (bool): if true, same layer applied to both nodes in edge
            return_alpha (bool): indicate whether to return computed attention
            device (DEVICE_TYPING): cpu or cuda
        """
        # initialize message passing with sum aggergation scheme
        super().__init__(aggr='add')

        # initialize layers
        share_weights = False
        self.lin_s = Linear(in_channels, out_channels, bias=False, device=device)
        self.lin_t = self.lin_s if share_weights else Linear(in_channels, out_channels, bias=False, device=device)
        self.attn = Linear(out_channels, 1, bias=False, device=device)
        self.bias = Parameter(torch.empty(out_channels, device=device))
        self.return_alpha = return_alpha
        self.alpha = None
        self.dp_rate = dp_rate
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bernoulli = False
        if self.bernoulli:
            input('Confirm Sigmoid Sampling in training?')
        else:
            print('No Bernoulli or Sigmoid Sampling!!!')
    
        self.reset_parameters()

    def reset_parameters(self):
        """ Reset model parameters """
        self.lin_s.reset_parameters()
        self.lin_t.reset_parameters()
        self.attn.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, dp_rate: float):
        """
        Conduct a round of message passing.

        Args:
            x (torch.Tensor): [N, in_channels]
            edge_index (torch.Tensor: [2, E]
        
        Returns:
            (torch.Tensor) [N, out_channels]
        """
        self.dp_rate = dp_rate
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = to_undirected(edge_index)
        og_edge_index = edge_index
        if self.dp_rate != 0:
            # edge_index, edge_mask = dropout_edge(edge_index, self.dp_rate)
            edge_index, edge_mask, _ = dropout_node(edge_index, self.dp_rate)
        # else:
        #     print('skipping dropout')

        # Get node embeddings
        x_s = self.lin_s(x)
        x_t = self.lin_t(x)

        # Start propagating message
        out = self.propagate(edge_index, x=(x_t, x_s))
        out += self.bias

        # Parse attention values
        if self.dp_rate != 0:
            alpha_out = torch.zeros(edge_mask.shape, device=self.device)
            alpha_out[edge_mask] = self.alpha
            self.alpha = alpha_out
            edge_index = og_edge_index
        self.alpha = self.alpha[torch.argsort(edge_index[1])]

        return (out, self.alpha) if self.return_alpha else out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index_i: torch.Tensor):
        """
        Compute attention for each node pair and construct messages

        Args:
            x_i (torch.Tensor): target (query) node features [E, out_channels]
            x_j (torch.Tensor): source (key) node features [E, out_channels]
            edge_index_i (torch.Tensor): matrix labeling neighboring nodes for computing softmax

        Returns:
            (torch.Tensor) messages from source j to query i with attention
        """
        # Compute attention
        e = self.attn(F.leaky_relu(x_i + x_j, negative_slope=0.2))
        a = softmax(e, index=edge_index_i)
        self.alpha = a.flatten()
        # TESTING BERNOULLI SAMPLING
        if self.bernoulli:
            a = torch.sigmoid((a - a.mean())*20) #torch.bernoulli(a)
        return x_j * a
