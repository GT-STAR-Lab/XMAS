import torch
import pyvirtualdisplay

# Tensordict modules
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor

# Data collection - modules to work with replay buffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import CompositeSpec

# Env
from torchrl.envs import RewardSum, TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from hetmamp.envs.transforms import ConnectivityGraph, RemoveLidar

from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import ProbabilisticActor, TanhNormal, MultiAgentMLP, MLP
from hetmamp.modules import MultiAgentGNN
# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
import io
import imageio
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from torch_geometric.explain import Explainer, GNNExplainer, DummyExplainer, GraphMaskExplainer, AttentionExplainer
from scipy import sparse
import numpy as np
from torch_geometric.data import Data

# devices
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print(device)
vmas_device = device

# Sampling
frames_per_batch = 6000*3
n_iters = 50*3 # number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30#*2 # number of optimization steps per training iteration
minibatch_size = 400*2 # Size of mini-batches in each optimization step
lr = 3e-4
max_grad_norm = 1.0

# PPO
clip_epsilon = 0.2 # clip value for PPO loss
gamma = 0.99999 # discount factor
lmbda = 0.9 # lambda for generalized advantage estimation
entropy_eps = 1e-4 # coefficient for entropy term in ppo loss

# Environmnet
max_steps = 150 # episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
) # number of vectorized envs
scenario_name = "navigation"
n_agents = int(input('How many agents?: ')) #3#+4

max_steps = max_steps + 250 if n_agents > 4 else max_steps
print('max steps: ', max_steps)

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    #Scenario kwargs
    n_agents=n_agents, # 
)

print("action_spec:", env.action_spec)
print("reward_spec:", env.reward_spec)
print("done_spec:", env.done_spec)
print("observation_spec:", env.observation_spec)

print("action_keys:", env.action_keys)
print("reward_keys:", env.reward_keys)
print("done_keys:", env.done_keys)
print("reset_keys:", env.reset_keys)


# Transform - sum up the rewards to get the episode return.
# this "meta-wraps" the environment
env = TransformedEnv(
    env,
    Compose(
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
        RemoveLidar(in_keys=[("agents", "observation")], 
                    out_keys=[("agents", "observation")],
                    ),
        ConnectivityGraph(in_keys=[("agents", "observation")], 
                          out_keys=[("agents", "edge_index")],
                          n_agents=n_agents,
                          conn_type='full',
                          ),
    )
)

# check_env_specs(env)


# Seeing a rollout using random actions
n_rollout_steps = 5
# rollout = env.rollout(n_rollout_steps)
# print("rollout of five steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)

share_parameters_policy = True

num_gnn_layers = 3-1 #2 Layer GAT works best empirically (note its desired gnn layers + 1)

use_gat_attn = True

alpha_ret = True

dp_rate = 0

# Communication Network
comm_net = MultiAgentGNN(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    gnn_agent_outputs=2 * env.action_spec.shape[-1],
    n_agent_outputs=2 * env.action_spec.shape[-1],
    n_agents = n_agents,
    share_params=share_parameters_policy,
    attention=use_gat_attn,
    return_alpha=alpha_ret,
    device=device,
    action_output=True,
    num_layers=num_gnn_layers,
    dp_rate=dp_rate
)

# Multi Agent MLP
# multi_agent_net = MultiAgentMLP(
#     n_agent_inputs=comm_net.n_agent_outputs,
#     n_agent_outputs=2 * env.action_spec.shape[-1], # 2 because mu and std,
#     n_agents = n_agents,
#     centralised=False,
#     share_params=share_parameters_policy,
#     device=device,
#     depth=2,
#     num_cells=256,
#     activation_class=torch.nn.Tanh
# )

# wrap a neural network in a TensorDictModule. Takes input in_keys, runs through NN, and outputs to out keys
# NOTE: This has to be wrapped in TensorDictSequential, Torch Sequential doesn't like multiple inputs :(
policy_module = TensorDictSequential(
    TensorDictModule(
        comm_net,
        in_keys=[("agents", "observation"), ("agents", "edge_index")],
        out_keys= ["comm_out", ("agents", "attn")],
    ),
    TensorDictModule(
        NormalParamExtractor(), # separates the last dimension into two outputs: a loc and a non-negative scale
        in_keys=["comm_out"],
        out_keys=[("agents", "loc"), ("agents", "scale")]
    )
)

# wrap TensorDictModule in a ProbabilisticActor
# builds a distribution out of mu and std
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.unbatched_action_spec[env.action_key].space.low,
        "max": env.unbatched_action_spec[env.action_key].space.high
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
) # we'll need to log-prob for the PPO loss

# Critic
share_parameters_critic = True
mappo = True

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")]
)

print("Running policy:", policy(env.reset()))
print("Running value:", critic(env.reset()))

# Data collector - collects rollouts using the policy
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames
)

# replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size
)

# Loss Function
loss_module = ClipPPOLoss(
    actor=policy,
    critic=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False, # important to avoid normalizing across the agent dimension
)
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated")
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
) # We build GAE

GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

# test env
test_env = VmasEnv(
    scenario=scenario_name,
    num_envs=1,
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    #Scenario kwargs
    n_agents=n_agents, # 
)

test_env = TransformedEnv(
    test_env,
    Compose(
        RewardSum(in_keys=[test_env.reward_key], out_keys=[("agents", "episode_reward")]),
        RemoveLidar(in_keys=[("agents", "observation")], 
                    out_keys=[("agents", "observation")],
                    ),
        ConnectivityGraph(in_keys=[("agents", "observation")], 
                          out_keys=[("agents", "edge_index")],
                          n_agents=n_agents,
                          conn_type='full',
                          ),
    )
)


# # Training loop
# pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
# episode_reward_mean_list = []
# no_coll_rate = []
# attn_loss = []
# loss = []
# success_rate = []
# avg_rollout = []
# outer_step_epoch = 0
# for tensordict_data in collector:
#     tensordict_data.set(
#         ("next", "agents", "done"),
#         tensordict_data.get(("next", "done")).unsqueeze(-1)
#         .expand(tensordict_data.get_item_shape(("next", env.reward_key)))
#     ) # Expance the done and terimanted to match the reward shape (expected for value estimator)

#     with torch.no_grad():
#         GAE(
#             tensordict_data,
#             params=loss_module.critic_params,
#             target_params=loss_module.target_critic_params
#         )
#     data_view = tensordict_data.reshape(-1) # flatten the batch size to shuffle data
#     replay_buffer.extend(data_view)

#     for _ in range(num_epochs):
#         attn_loss_total = 0
#         loss_total = 0
#         for _ in range(frames_per_batch // minibatch_size):
#             subdata=replay_buffer.sample()
#             loss_vals=loss_module(subdata)

#             loss_value = (
#                 loss_vals["loss_objective"]
#                 + loss_vals["loss_critic"]
#                 + loss_vals["loss_entropy"]
#                 + loss_vals["loss_attn"]
#             )

#             # loss_value.requires_grad = True
#             attn_loss_total = attn_loss_total + loss_vals["loss_attn"].item()
#             loss_total = loss_total + loss_value.item()

#             loss_value.backward()

#             torch.nn.utils.clip_grad_norm_(
#                 loss_module.parameters(), max_grad_norm
#             )

#             optim.step()
#             optim.zero_grad()
#         attn_loss.append(attn_loss_total/(frames_per_batch // minibatch_size))
#         loss.append(loss_total/(frames_per_batch // minibatch_size))
#     collector.update_policy_weights_()

#     print(loss_vals["loss_attn"])

#     torch.save(collector.policy.module[0].module[0].module.state_dict(), 'model' + str(outer_step_epoch) + '.pth')
#     outer_step_epoch = outer_step_epoch + 1

#     # Logging
#     done = tensordict_data.get(("next", "agents", "done"))
#     episode_reward_mean = (
#         tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
#     )
#     episode_reward_mean_list.append(episode_reward_mean)
#     # success_rate.append(float(done.sum()/100.0))
#     pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
#     pbar.update()
#     rollout_count = []
#     success_count = 0
#     no_coll_count = 0
#     NUM_EVAL = 10
#     policy.module[0].module[0].module.agent_networks[0].dp_rate = 0
#     for _ in range(NUM_EVAL):
#         x = test_env.rollout(
#             max_steps=max_steps,
#             policy=policy,
#             auto_cast_to_device=True,
#             break_when_any_done=True,
#         )
#         rollout_count.append(x.batch_size[1])
#         if x.batch_size[1] < max_steps: #and (x['agents']['info']['agent_collisions'].sum() == 0).item():
#             success_count += 1
#         if (x['agents']['info']['agent_collisions'].sum() == 0).item():
#             no_coll_count += 1

#     avg_rollout.append(sum(rollout_count)/len(rollout_count))
#     success_rate.append(float(success_count/NUM_EVAL))
#     no_coll_rate.append(float(no_coll_count/NUM_EVAL))
#     policy.module[0].module[0].module.agent_networks[0].dp_rate = dp_rate # np.abs(np.random.triangular(-0.5, 0, 0.5))
#     print(avg_rollout[-1])
#     print(success_rate[-1])
#     print(no_coll_rate[-1])

# print('generating plots')
# plt.plot(episode_reward_mean_list)
# plt.xlabel("Training iterations")
# plt.ylabel("Reward")
# plt.title("Episode reward mean")
# plt.savefig('reward.png')
# plt.close()

# print(success_rate)
# plt.plot(success_rate)
# plt.xlabel("Training iterations")
# plt.ylabel("Success Rate")
# plt.title("Success rate mean")
# plt.savefig('success.png')
# plt.close()

# print(no_coll_rate)
# plt.plot(no_coll_rate)
# plt.xlabel("Training iterations")
# plt.ylabel("No Collision Rate")
# plt.title("No Collision rate mean")
# plt.savefig('collision.png')
# plt.close()

# plt.plot(avg_rollout)
# plt.xlabel("Training iterations")
# plt.ylabel("Makespan")
# plt.title("Makespan mean")
# plt.savefig('makespan.png')
# plt.close()

# plt.plot(attn_loss)
# plt.xlabel("Training iterations")
# plt.ylabel("Attention Loss Term")
# plt.title("Attention Loss")
# plt.savefig('attn_loss.png')
# plt.close()

# plt.plot(loss)
# plt.xlabel("Training iterations")
# plt.ylabel("Loss")
# plt.title("Loss mean")
# plt.savefig('loss.png')
# plt.close()

# # import copy

# # trained_policy_module = copy.deepcopy(collector.policy.module[0].module[0].module)

# # trained_policy_module.agent_networks[0].explain_yourself = True

# # collector.policy.module[0].module[0].module.agent_networks[0].explain_yourself = True

# import os

# try:
#     os.remove('model.pth')
# except:
#     pass

# torch.save(collector.policy.module[0].module[0].module.state_dict(), 'model.pth')

# print('saving module andl loading to avoid deepcopy issues')
# import time
# time.sleep(5)

trained_policy_module = MultiAgentGNN(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    gnn_agent_outputs=2 * env.action_spec.shape[-1],
    n_agent_outputs=2 * env.action_spec.shape[-1],
    n_agents = n_agents,
    share_params=share_parameters_policy,
    attention=use_gat_attn,
    return_alpha=False,
    device=device,
    action_output=True,
    num_layers=num_gnn_layers,
    dp_rate=0
)
trained_policy_module.load_state_dict(torch.load('model.pth'))

trained_policy_module.agent_networks[0].explain_yourself = False

trained_policy_module.agent_networks[0].dp_rate = 0

policy_module = TensorDictSequential(
    TensorDictModule(
        trained_policy_module,
        in_keys=[("agents", "observation"), ("agents", "edge_index")],
        out_keys= ["comm_out", ("agents", "attn")],
    ),
    TensorDictModule(
        NormalParamExtractor(), # separates the last dimension into two outputs: a loc and a non-negative scale
        in_keys=["comm_out"],
        out_keys=[("agents", "loc"), ("agents", "scale")]
    )
)

# wrap TensorDictModule in a ProbabilisticActor
# builds a distribution out of mu and std
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.unbatched_action_spec[env.action_key].space.low,
        "max": env.unbatched_action_spec[env.action_key].space.high
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob"),
) # we'll need to log-prob for the PPO loss

# FOR THE EXPLAINER
exp_policy_module = MultiAgentGNN(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    gnn_agent_outputs=2 * env.action_spec.shape[-1],
    n_agent_outputs=2 * env.action_spec.shape[-1],
    n_agents = n_agents,
    share_params=share_parameters_policy,
    attention=use_gat_attn,
    return_alpha=False,
    device=device,
    action_output=True,
    num_layers=num_gnn_layers,
    dp_rate=0
)
exp_policy_module.load_state_dict(torch.load('model.pth'))

exp_policy_module.agent_networks[0].explain_yourself = True

exp_policy_module.agent_networks[0].dp_rate = 0

explainers = []

# explainer_object = Explainer(
#     model=exp_policy_module,
#     algorithm=GNNExplainer(epochs=300, lr=0.02),
#     explanation_type='model',
#     node_mask_type='object',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='regression',
#         task_level='node'
#     ),
# )

# # GRAPHMASK
# explainer_features = Explainer(
#     model=exp_policy_module,
#     algorithm=GraphMaskExplainer(num_gnn_layers-1, epochs=250),
#     explanation_type='model',
#     node_mask_type=None,
#     edge_mask_type='object',
#     model_config=dict(
#         mode='regression',
#         task_level='graph'
#     ),
# )
# use  below one for GraphMask
explainer_features = Explainer(
    model=exp_policy_module,
    algorithm=GraphMaskExplainer(num_gnn_layers-1, epochs=250),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node'
    ),
)
explainers.append(GraphMaskExplainer(num_gnn_layers-1))

#ATTN EXP
explainer_features = Explainer(
    model=exp_policy_module,
    algorithm=AttentionExplainer(reduce='mean'),
    explanation_type='model',
    node_mask_type=None,
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node'
    ),
)
explainers.append(AttentionExplainer(reduce='mean'))

#GNNEXP
# explainer_features = Explainer(
#     model=exp_policy_module,
#     algorithm=GNNExplainer(),
#     explanation_type='model',
#     node_mask_type=None,
#     edge_mask_type='object',
#     model_config=dict(
#         mode='regression',
#         task_level='graph'
#     ),
# )

# use below one for GNNEXP
explainer_features = Explainer(
    model=exp_policy_module,
    algorithm=GNNExplainer(),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node'
    ),
)
explainers.append(GNNExplainer())

#DUMMY EXP
explainer_features = Explainer(
    model=exp_policy_module,
    algorithm=DummyExplainer(),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node'
    ),
)
explainers.append(DummyExplainer())

explainer_features = None

# explanation = explainer(data.x, data.edge_index)
rollout_iter = 0
# f = open(scenario_name + "_object.txt", "w")
# f.close()
# f = open(scenario_name + "_feature.txt", "w")
# f.close()
fid_plus = []
fid_minus = []
char_scores = []
delta_fid = []
unfaithfulness = []
mod_unfaith = []
sparsity = []
stability = []
prev_edge_mask = None
prev_edge_index = None
from torch_geometric.explain import metric
from statistics import mean, stdev

import csv
from itertools import zip_longest

normParamExtract = NormalParamExtractor()

# render the latest rollout and save it to a gif
def rendering_callback(env, td):
    global rollout_iter
    global fid_plus
    global fid_minus
    global char_scores
    global delta_fid
    global explainer_features
    global normParamExtract
    global trained_policy_module
    global unfaithfulness
    global mod_unfaith
    global sparsity
    global stability
    global prev_edge_mask
    global prev_edge_index
    # env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
    # coo = sparse.coo_array(np.triu(td['agents']['edge_index'][0][0].cpu().numpy()))
    # data = Data(x=td['agents']['observation'][0], edge_index=torch.Tensor(np.vstack((coo.row, coo.col))))
    # data = Data(x=td['agents']['observation'][0], edge_index=torch.Tensor(np.vstack((coo.row, coo.col))).type(torch.int8).to(device))
    # data = Data(x=td['agents']['observation'][0], edge_index=td['agents']['edge_index'][0,:,0].to_sparse().indices())
    data = Data(x=td['agents']['observation'][0], edge_index=td['agents']['edge_index'][0][0].to_sparse().indices())
    # explanation_object = explainer_object(data.x, data.edge_index)
    explanation_features = explainer_features(data.x, data.edge_index)

    fp, fm = metric.fidelity(explainer=explainer_features, explanation=explanation_features)
    fid_plus.append(fp)
    fid_minus.append(fm)
    char_scores.append(metric.characterization_score(fp, fm))
    delta_fid.append(fp-fm)

    node_mask = explanation_features.get('node_mask')
    edge_mask = explanation_features.get('edge_mask')
    loc = explainer_features.get_masked_prediction(
        explanation_features.x,
        explanation_features.edge_index,
        node_mask,
        edge_mask,
        unfaithful_save_og=True)
    loc2, scale2 = normParamExtract(explainer_features.model.agent_networks[0].og_output)
    loc1, scale1 = normParamExtract(trained_policy_module(x=td['agents']['observation'][0], edge_index=td['agents']['edge_index'][0][0].to_sparse().indices()))
    kl = 0.5*(((loc2 - loc1)**2)/(scale2) + (scale1)/(scale2) - torch.log((scale1)/(scale2)) - 1)
    unfaithfulness.append(torch.mean(1 - torch.exp(-kl)).item())

    loc1, scale1, loc2, scale2 = loc2, scale2, loc1, scale1
    kl2 = 0.5*(((loc2 - loc1)**2)/(scale2) + (scale1)/(scale2) - torch.log((scale1)/(scale2)) - 1)

    j_dist = kl + kl2
    mod_unfaith.append(torch.mean(1 - torch.exp(-j_dist)).item())

    sparsity.append(1 - sum(explanation_features.edge_mask > 0.5).item()/len(explanation_features.edge_mask))

    if prev_edge_mask is not None and prev_edge_index is not None and torch.allclose(prev_edge_index, explanation_features.edge_index):
        stability.append(torch.linalg.vector_norm(explanation_features.edge_mask - prev_edge_mask).item())
    prev_edge_mask = explanation_features.edge_mask
    prev_edge_index = explanation_features.edge_index


    env.base_env.scenario.edge_index = explanation_features.edge_index
    env.base_env.scenario.edge_mask = explanation_features.get('edge_mask')

    # f = open(scenario_name + "_edge.txt", "a")
    
    # edge_mask_str = 'edge_mask'
    # node_mask_str = 'node_mask'
    # f.write(f'Edge mask for iter {rollout_iter}:\n {explanation_features.edge_index}\n {explanation_features.get(edge_mask_str)} \n')
    # f.write(f'Node mask for iter {rollout_iter}:\n {explanation_features.x}\n {explanation_features.get(node_mask_str)} \n')
    # f.close()

    # env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
    try:
        after_render = env.render(mode="rgb_array")
        before_render = env.before_render(mode="rgb_array")

        env.frames.append(Image.fromarray(after_render))
        env.frames_before.append(Image.fromarray(before_render))
    except:
        pass

    # f = open(scenario_name + "_object.txt", "a")
    # f.write(f'Node mask for iter {rollout_iter}:\n {explanation_object.node_mask} \n')
    # f.close()

    # f = open(scenario_name + "_feature.txt", "a")
    # f.write(f'Feature mask for iter {rollout_iter}:\n {explanation_features.node_mask} \n')
    # f.close()
    # path = scenario_name + 'feature_importance' + str(rollout_iter) + '.png'
    # explanation_features.visualize_feature_importance(path)
    path = scenario_name + 'graph_vis' + str(rollout_iter) + '.png'
    # explanation_features.visualize_graph(path, env=env)
    rollout_iter += 1
    if 'attn' in td.keys():
        attn.append(td['attn'][0])

scenarios = ["navigation"]
total_num_experiments = 50
csv_file_name = "eval_metrics_rand_trials_envs_" + str(total_num_experiments) + ".csv"
f = open(csv_file_name, "w")
f.write(f'Scenario,Explainer,FidPlus,FidMinus,DeltaFid,CharScore,Unfaithfulness,Sparsity,Stability,stdevFidPlus,stdevFidMinus,stdevDeltaFid,stdevCharScore,stdevUnfaithfulness,stdevSparsity,stdevStability,lenFidPlus,lenFidMinus,lenDeltaFid,lenCharScore,lenUnfaithfulness,lenSparsity,lenStability,MeanFidPlus,MeanFidMinus,MeanDeltaFid,MeanCharScore,MeanUnfaithfulness,MeanSparsity,MeanStability,stdevMeanFidPlus,stdevMeanFidMinus,stdevMeanDeltaFid,stdevMeanCharScore,stdevMeanUnfaithfulness,stdevMeanSparsity,stdevMeanStability,lenMeanFidPlus,lenMeanFidMinus,lenMeanDeltaFid,lenMeanCharScore,lenMeanUnfaithfulness,lenMeanSparsity,lenMeanStability\n')
f.close()
# scenario_name = "navigation_scenario1"

episode_reward_mean_list = []
rollout_count = []
success_count = []
new_success_count = []
no_coll_count = []
coll_rate = []
NUM_EVAL = 0

for explainer in explainers:
    # FOR THE EXPLAINER
    exp_policy_module = MultiAgentGNN(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        gnn_agent_outputs=2 * env.action_spec.shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents = n_agents,
        share_params=share_parameters_policy,
        attention=use_gat_attn,
        return_alpha=False,
        device=device,
        action_output=True,
        num_layers=num_gnn_layers,
        dp_rate=0
    )
    exp_policy_module.load_state_dict(torch.load('model.pth'))

    exp_policy_module.agent_networks[0].explain_yourself = True

    exp_policy_module.agent_networks[0].dp_rate = 0
    if isinstance(explainer, AttentionExplainer):
        explainer_features = Explainer(
            model=exp_policy_module,
            algorithm=explainer,
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node'
            ),
        )
    else:
        explainer_features = Explainer(
            model=exp_policy_module,
            algorithm=explainer,
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node'
            ),
        )
    for scenario_name in scenarios:

        f = open(scenario_name + "_object.txt", "w")
        f.close()
        f = open(scenario_name + "_feature.txt", "w")
        f.close()
        f = open(scenario_name + "_edge.txt", "w")
        f.close()

        agg_fid_plus = []
        agg_fid_minus = []
        agg_delta_fid = []
        agg_char_scores = []
        agg_unfaithfulness = []
        agg_mod_unfaith = []
        agg_sparsity = []
        agg_stability = []

        agg_mean_fid_plus = []
        agg_mean_fid_minus = []
        agg_mean_delta_fid = []
        agg_mean_char_scores = []
        agg_mean_unfaithfulness = []
        agg_mean_mod_unfaith = []
        agg_mean_sparsity = []
        agg_mean_stability = []

        #same envs

        torch.manual_seed(0)
        import random
        random.seed(0)
        np.random.seed(0)

        for num_experiments in range(total_num_experiments):

            env = VmasEnv(
                scenario=scenario_name,
                num_envs=1,
                continuous_actions=True,
                max_steps=max_steps,
                device=vmas_device,
                #Scenario kwargs
                n_agents=n_agents, # 
            )

            env = TransformedEnv(
                env,
                Compose(
                    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
                    RemoveLidar(in_keys=[("agents", "observation")], 
                                out_keys=[("agents", "observation")],
                                ),
                    ConnectivityGraph(in_keys=[("agents", "observation")], 
                                    out_keys=[("agents", "edge_index")],
                                    n_agents=n_agents,
                                    conn_type='full',
                                    ),
                )
            )

            env.frames = []
            env.frames_before = []
            attn = []
            rollout_iter = 0
            fid_plus = []
            fid_minus = []
            char_scores = []
            delta_fid = []
            unfaithfulness = []
            mod_unfaith = []
            sparsity = []
            stability = []
            prev_edge_mask = None
            prev_edge_index = None


            # with torch.no_grad():
            torch.set_grad_enabled(True)
            x = env.rollout(
                max_steps=max_steps,
                policy=policy,
                callback=rendering_callback,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )
            NUM_EVAL += 1
            done = x.get(("next", "done")).unsqueeze(-1).expand(x.get_item_shape(("next", env.reward_key)))
            episode_reward_mean = (
                x.get(("next", "agents", "episode_reward"))[done].mean().item()
            )
            episode_reward_mean_list.append(episode_reward_mean)
            rollout_count.append(x.batch_size[1])
            # if (x['agents']['info']['collision_rew'].sum()).item() == 0:
            #     no_coll_count.append(1)
            if (x['agents']['info']['agent_collisions'].sum() == 0).item():
                no_coll_count.append(1)
            else:
                no_coll_count.append(0)
            if x.batch_size[1] < max_steps:
                success_count.append(1)
                new_success_count.append(1)
            else:
                success_count.append(0)
                new_success_count.append(((abs(x['agents']['observation'][0][-1][:,-2:]) < 0.05).sum()/(n_agents*2)).item())
            coll_rate.append(((x['agents']['info']['agent_collisions'].sum().abs()/2)/max(x['agents']['info']['agent_collisions'].shape)).item())
            # try:
            #     env.frames[0].save(
            #         f"{scenario_name}_after.gif",
            #         save_all=True,
            #         append_images=env.frames[1:],
            #         duration=1000,
            #         loop=0,
            #     )
            #     env.frames_before[0].save(
            #         f"{scenario_name}_before.gif",
            #         save_all=True,
            #         append_images=env.frames_before[1:],
            #         duration=1000,
            #         loop=0,
            #     )
            # except:
            #     print('error in render, passing')
            
            # for i in range(min(len(env.frames_before), len(env.frames))):
            #     env.frames[i].save(f"{scenario_name}_after" + str(i) + '.png')
            #     env.frames_before[i].save(f"{scenario_name}_before" + str(i) + '.png')

            # fp_in = scenario_name + "graph_vis*.png"
            # fp_out = scenario_name + "graph_vis.gif"

            # import glob
            # # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
            # img.save(fp=fp_out, format='GIF', append_images=imgs,
            #         save_all=True, duration=1000, loop=0)
            
            # print('generating plot')
            # plt.plot(fid_plus)
            # plt.xlabel("Steps")
            # plt.ylabel("Pos Fid")
            # plt.title("Positive Fidelity")
            # plt.savefig(scenario_name + str(num_experiments) + '_posfid.png')
            # plt.close()

            # print('generating plot')
            # plt.plot(fid_minus)
            # plt.xlabel("Steps")
            # plt.ylabel("Neg Fid")
            # plt.title("Negative Fidelity")
            # plt.savefig(scenario_name + str(num_experiments) + '_negfid.png')
            # plt.close()

            # print('generating plot')
            # plt.plot(char_scores)
            # plt.xlabel("Steps")
            # plt.ylabel("Char Score")
            # plt.title("Characterization Score")
            # plt.savefig(scenario_name + str(num_experiments) + '_char.png')
            # plt.close()

            # print('generating plot')
            # plt.plot(delta_fid)
            # plt.xlabel("Steps")
            # plt.ylabel("Delta Fid")
            # plt.title("Delta Fidelity Score")
            # plt.savefig(scenario_name + str(num_experiments) + '_deltafid.png')
            # plt.close()

            agg_fid_plus += fid_plus
            agg_fid_minus += fid_minus
            agg_char_scores += char_scores
            agg_delta_fid += delta_fid
            agg_unfaithfulness += unfaithfulness
            agg_mod_unfaith += mod_unfaith
            agg_sparsity += sparsity
            agg_stability += stability

            agg_mean_fid_plus.append(mean(fid_plus))
            agg_mean_fid_minus.append(mean(fid_minus))
            agg_mean_char_scores.append(mean(char_scores))
            agg_mean_delta_fid.append(mean(delta_fid))
            agg_mean_unfaithfulness.append(mean(unfaithfulness))
            agg_mean_mod_unfaith.append(mean(mod_unfaith))
            agg_mean_sparsity.append(mean(sparsity))
            agg_mean_stability.append(mean(stability))

        f = open(csv_file_name, "a")
        f.write(f'{scenario_name},{explainer.__class__.__name__},{mean(agg_fid_plus)},{mean(agg_fid_minus)},{mean(agg_delta_fid)},{mean(agg_char_scores)},{mean(agg_unfaithfulness)},{mean(agg_sparsity)},{mean(agg_stability)},{stdev(agg_fid_plus)},{stdev(agg_fid_minus)},{stdev(agg_delta_fid)},{stdev(agg_char_scores)},{stdev(agg_unfaithfulness)},{stdev(agg_sparsity)},{stdev(agg_stability)},{len(agg_fid_plus)},{len(agg_fid_minus)},{len(agg_delta_fid)},{len(agg_char_scores)},{len(agg_unfaithfulness)},{len(agg_sparsity)},{len(agg_stability)},{mean(agg_mean_fid_plus)},{mean(agg_mean_fid_minus)},{mean(agg_mean_delta_fid)},{mean(agg_mean_char_scores)},{mean(agg_mean_unfaithfulness)},{mean(agg_mean_sparsity)},{mean(agg_mean_stability)},{stdev(agg_mean_fid_plus)},{stdev(agg_mean_fid_minus)},{stdev(agg_mean_delta_fid)},{stdev(agg_mean_char_scores)},{stdev(agg_mean_unfaithfulness)},{stdev(agg_mean_sparsity)},{stdev(agg_mean_stability)},{len(agg_mean_fid_plus)},{len(agg_mean_fid_minus)},{len(agg_mean_delta_fid)},{len(agg_mean_char_scores)},{len(agg_mean_unfaithfulness)},{len(agg_mean_sparsity)},{len(agg_mean_stability)}\n')
        # f.write(f'Scenario:\n {scenario_name}\n')
        # f.write(f'Fid Plus:\n {mean(fid_plus)}\n')
        # f.write(f'Fid Minus:\n {mean(fid_minus)}\n')
        # f.write(f'Char Score:\n {mean(char_scores)}\n')
        # f.write(f'Delta Fid:\n {mean(delta_fid)}\n')
        f.close()

        d = [agg_fid_plus, agg_fid_minus, agg_delta_fid, agg_unfaithfulness, agg_sparsity, agg_stability, agg_mod_unfaith]
        export_data = zip_longest(*d, fillvalue = '')
        with open(str(explainer.__class__.__name__) + '_numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("FidPlus", "FidMinus", "DeltaFid", "Unfaithfulness", "Sparsity", "Stability", "SymmetricUnfaithfulness"))
            wr.writerows(export_data)
        myfile.close()

d = [episode_reward_mean_list, rollout_count, success_count, no_coll_count, coll_rate, new_success_count]
export_data = zip_longest(*d, fillvalue = '')
with open('performance_numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("EpisodeReward", "Makespan", "Success Rate", "No Collisions Rate", "Collision Rate", "New Success Rate"))
    wr.writerows(export_data)
myfile.close()

print('Episode Reward Mean: ', mean(episode_reward_mean_list))
print('Episode Reward StdDev: ', stdev(episode_reward_mean_list))

print('Makespan Mean: ', mean(rollout_count))
print('Makespan StdDev: ', stdev(rollout_count))

print('Success Rate Mean: ', mean(success_count))
print('Success Rate StdDev: ', stdev(success_count))

print('No Collision Rate Mean: ', mean(no_coll_count))
print('No Collision Rate StdDev: ', stdev(no_coll_count))

print('Collision Rate Mean: ', mean(coll_rate))
print('Collision Rate StdDev: ', stdev(coll_rate))

print('New Success Rate Mean: ', mean(new_success_count))
print('New Success Rate StdDev: ', stdev(new_success_count))





if attn:
    def plot_attn_matrix(matrix):
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
        plt.title("Attention")
        
        # Save attention matrix as an in-memory image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        buffer.seek(0)
        return imageio.imread(buffer)

    # Create heatmaps and store them as in-memory images
    attn_maps = [plot_attn_matrix(matrix.cpu().numpy()) for matrix in attn]

    # Create a GIF
    with imageio.get_writer(f'{scenario_name}_attention.gif', mode='I', duration=10, loops=0) as writer:
        for image in attn_maps:
            writer.append_data(image)

# from IPython.display import Image
# filename = f"{scenario_name}.gif"
# print(filename)
# Image(open(filename, "rb").read())
