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
from hetmamp.envs.transforms import ConnectivityGraph

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

from torch_geometric.explain import Explainer, GNNExplainer, DummyExplainer, GraphMaskExplainer
from scipy import sparse
import numpy as np
from torch_geometric.data import Data

# devices
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print(device)
vmas_device = device

# Sampling
frames_per_batch = 10000
n_iters = 1000 # number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 5#*2 # number of optimization steps per training iteration
minibatch_size = 10000 # Size of mini-batches in each optimization step
lr = 7e-4
max_grad_norm = 10

# PPO
clip_epsilon = 0.05 # clip value for PPO loss
gamma = 0.9999 # discount factor
lmbda = 0.95 # lambda for generalized advantage estimation
entropy_eps = 1e-4 # coefficient for entropy term in ppo loss

# Environmnet
max_steps = 500 # episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
) # number of vectorized envs
scenario_name = "discovery"
n_agents = int(input('How many agents?: ')) #2+1+1
print('num agents: ', n_agents)

attn_ent_weight = 50
print('attn ent weight: ', attn_ent_weight)

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
        # RemoveLidar(in_keys=[("agents", "observation")], 
        #             out_keys=[("agents", "observation")],
        #             ),
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
    dp_rate=dp_rate,
    discovery=True
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
    depth=2,
    num_cells=128,
    activation_class=torch.nn.ReLU
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
    normalize_advantage=True, # important to avoid normalizing across the agent dimension
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
        # RemoveLidar(in_keys=[("agents", "observation")], 
        #             out_keys=[("agents", "observation")],
        #             ),
        ConnectivityGraph(in_keys=[("agents", "observation")], 
                          out_keys=[("agents", "edge_index")],
                          n_agents=n_agents,
                          conn_type='full',
                          ),
    )
)


# Training loop
pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
episode_reward_mean_list = []
no_coll_rate = []
attn_loss = []
loss = []
success_rate = []
avg_rollout = []
outer_step_epoch = 0
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done")).unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key)))
    ) # Expance the done and terimanted to match the reward shape (expected for value estimator)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_params,
            target_params=loss_module.target_critic_params
        )
    data_view = tensordict_data.reshape(-1) # flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        attn_loss_total = 0
        loss_total = 0
        for _ in range(frames_per_batch // minibatch_size):
            subdata=replay_buffer.sample()
            loss_vals=loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
                + attn_ent_weight*loss_vals["loss_attn"]
            )

            # loss_value.requires_grad = True
            attn_loss_total = attn_loss_total + loss_vals["loss_attn"].item()
            loss_total = loss_total + loss_value.item()

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )

            optim.step()
            optim.zero_grad()
        attn_loss.append(attn_loss_total/(frames_per_batch // minibatch_size))
        loss.append(loss_total/(frames_per_batch // minibatch_size))
    collector.update_policy_weights_()

    print(loss_vals["loss_attn"])

    torch.save(collector.policy.module[0].module[0].module.state_dict(), 'model' + str(outer_step_epoch) + '.pth')
    outer_step_epoch = outer_step_epoch + 1

    # Logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    # success_rate.append(float(done.sum()/100.0))
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()
    rollout_count = []
    success_count = 0
    no_coll_count = 0
    NUM_EVAL = 10
    policy.module[0].module[0].module.agent_networks[0].dp_rate = 0
    for _ in range(NUM_EVAL):
        x = test_env.rollout(
            max_steps=max_steps,
            policy=policy,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )
        rollout_count.append(x.batch_size[1])
        if x.batch_size[1] < max_steps: #and (x['agents']['info']['agent_collisions'].sum() == 0).item():
            success_count += 1
        # if (x['agents']['info']['agent_collisions'].sum() == 0).item():
        #     no_coll_count += 1

    avg_rollout.append(sum(rollout_count)/len(rollout_count))
    success_rate.append(float(success_count/NUM_EVAL))
    # no_coll_rate.append(float(no_coll_count/NUM_EVAL))
    policy.module[0].module[0].module.agent_networks[0].dp_rate = dp_rate # np.abs(np.random.triangular(-0.5, 0, 0.5))
    print(avg_rollout[-1])
    print(success_rate[-1])
    # print(no_coll_rate[-1])

print('generating plots')
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.savefig('reward.png')
plt.close()

print(success_rate)
plt.plot(success_rate)
plt.xlabel("Training iterations")
plt.ylabel("Success Rate")
plt.title("Success rate mean")
plt.savefig('success.png')
plt.close()

# print(no_coll_rate)
# plt.plot(no_coll_rate)
# plt.xlabel("Training iterations")
# plt.ylabel("No Collision Rate")
# plt.title("No Collision rate mean")
# plt.savefig('collision.png')
# plt.close()

plt.plot(avg_rollout)
plt.xlabel("Training iterations")
plt.ylabel("Makespan")
plt.title("Makespan mean")
plt.savefig('makespan.png')
plt.close()

plt.plot(attn_loss)
plt.xlabel("Training iterations")
plt.ylabel("Attention Loss Term")
plt.title("Attention Loss")
plt.savefig('attn_loss.png')
plt.close()

plt.plot(loss)
plt.xlabel("Training iterations")
plt.ylabel("Loss")
plt.title("Loss mean")
plt.savefig('loss.png')
plt.close()

# import copy

# trained_policy_module = copy.deepcopy(collector.policy.module[0].module[0].module)

# trained_policy_module.agent_networks[0].explain_yourself = True

# collector.policy.module[0].module[0].module.agent_networks[0].explain_yourself = True

import os

try:
    os.remove('model.pth')
except:
    pass

torch.save(collector.policy.module[0].module[0].module.state_dict(), 'model.pth')

print('saving module andl loading to avoid deepcopy issues')
import time
time.sleep(5)

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
    dp_rate=0,
    discovery=True
)
trained_policy_module.load_state_dict(torch.load('model.pth'))

trained_policy_module.agent_networks[0].explain_yourself = True

trained_policy_module.agent_networks[0].dp_rate = 0

# explainer_object = Explainer(
#     model=trained_policy_module,
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
explainer_features = Explainer(
    model=trained_policy_module,
    algorithm=GraphMaskExplainer(num_gnn_layers-1, epochs=250),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node'
    ),
)

#GNNEXP
# explainer_features = Explainer(
#     model=trained_policy_module,
#     algorithm=GNNExplainer(),
#     explanation_type='model',
#     node_mask_type='object',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='regression',
#         task_level='node'
#     ),
# )

# explanation = explainer(data.x, data.edge_index)
rollout_iter = 0
# f = open(scenario_name + "_object.txt", "w")
# f.close()
# f = open(scenario_name + "_feature.txt", "w")
# f.close()

# render the latest rollout and save it to a gif
def rendering_callback(env, td):
    global rollout_iter
    # env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
    # coo = sparse.coo_array(np.triu(td['agents']['edge_index'][0][0].cpu().numpy()))
    # data = Data(x=td['agents']['observation'][0], edge_index=torch.Tensor(np.vstack((coo.row, coo.col))))
    # data = Data(x=td['agents']['observation'][0], edge_index=torch.Tensor(np.vstack((coo.row, coo.col))).type(torch.int8).to(device))
    # data = Data(x=td['agents']['observation'][0], edge_index=td['agents']['edge_index'][0,:,0].to_sparse().indices())
    data = Data(x=td['agents']['observation'][0], edge_index=td['agents']['edge_index'][0][0].to_sparse().indices())
    # explanation_object = explainer_object(data.x, data.edge_index)
    explanation_features = explainer_features(data.x, data.edge_index)

    env.base_env.scenario.edge_index = explanation_features.edge_index
    env.base_env.scenario.edge_mask = explanation_features.get('edge_mask')

    f = open(scenario_name + "_edge.txt", "a")
    
    edge_mask_str = 'edge_mask'
    node_mask_str = 'node_mask'
    f.write(f'Edge mask for iter {rollout_iter}:\n {explanation_features.edge_index}\n {explanation_features.get(edge_mask_str)} \n')
    f.write(f'Node mask for iter {rollout_iter}:\n {explanation_features.x}\n {explanation_features.get(node_mask_str)} \n')
    f.close()

    # env.frames.append(Image.fromarray(env.render(mode="rgb_array")))
    after_render = env.render(mode="rgb_array")
    before_render = env.before_render(mode="rgb_array")

    env.frames.append(Image.fromarray(after_render))
    env.frames_before.append(Image.fromarray(before_render))

    # f = open(scenario_name + "_object.txt", "a")
    # f.write(f'Node mask for iter {rollout_iter}:\n {explanation_object.node_mask} \n')
    # f.close()

    # f = open(scenario_name + "_feature.txt", "a")
    # f.write(f'Feature mask for iter {rollout_iter}:\n {explanation_features.node_mask} \n')
    # f.close()
    # path = scenario_name + 'feature_importance' + str(rollout_iter) + '.png'
    # explanation_features.visualize_feature_importance(path)
    # path = scenario_name + 'graph_vis' + str(rollout_iter) + '.png'
    # explanation_features.visualize_graph(path, env=env)
    rollout_iter += 1
    if 'attn' in td.keys():
        attn.append(td['attn'][0])

scenarios = ["discovery"]
# scenario_name = "navigation_scenario1"
for scenario_name in scenarios:

    f = open(scenario_name + "_object.txt", "w")
    f.close()
    f = open(scenario_name + "_feature.txt", "w")
    f.close()
    f = open(scenario_name + "_edge.txt", "w")
    f.close()

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
            # RemoveLidar(in_keys=[("agents", "observation")], 
            #             out_keys=[("agents", "observation")],
            #             ),
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

    # with torch.no_grad():
    torch.set_grad_enabled(True)
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=rendering_callback,
        auto_cast_to_device=True,
        break_when_any_done=True,
    )
    env.frames[0].save(
        f"{scenario_name}_after.gif",
        save_all=True,
        append_images=env.frames[1:],
        duration=1000,
        loop=0,
    )
    env.frames_before[0].save(
        f"{scenario_name}_before.gif",
        save_all=True,
        append_images=env.frames_before[1:],
        duration=1000,
        loop=0,
    )
    for i in range(min(len(env.frames_before), len(env.frames))):
        env.frames[i].save(f"{scenario_name}_after" + str(i) + '.png')
        env.frames_before[i].save(f"{scenario_name}_before" + str(i) + '.png')

    # fp_in = scenario_name + "graph_vis*.png"
    # fp_out = scenario_name + "graph_vis.gif"

    # import glob
    # # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs,
    #         save_all=True, duration=1000, loop=0)

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
