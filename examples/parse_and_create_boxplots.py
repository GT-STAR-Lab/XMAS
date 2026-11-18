import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean, std
import os

import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 13})

dirs = ['no_attn_ent', 'attn_ent']
cdfs = []

import itertools
box_pairs_starter = list(itertools.product(['GraphMaskExplainer', 'GNNExplainer', 'AttentionExplainer'], dirs))
box_pairs = []
box_pairs_starter = list(itertools.product(box_pairs_starter, box_pairs_starter))
for (box1, box2) in box_pairs_starter:
    if box1 == box2 or (box1, box2) in box_pairs or (box2, box1) in box_pairs:
        continue
    elif box1[0] == box2[0]:
        box_pairs.append((box1, box2))
    elif box1[1] == box2[1]:
        box_pairs.append((box1, box2))
print(box_pairs)

cwd = os.getcwd()
task = None
num_agents = None

if 'navigation' in cwd:
    task = 'Navigation'
elif 'passage' in cwd:
    task = 'Passage'
elif 'discovery' in cwd:
    task = 'Discovery'
else:
    raise Exception

if '3_agent' in cwd:
    num_agents = '3 Agent Team'
elif '4_agent' in cwd:
    num_agents = '4 Agent Team'
elif '5_agent' in cwd:
    num_agents = '5 Agent Team'
elif '6_agent' in cwd:
    num_agents = '6 Agent Team'
elif '7_agent' in cwd:
    num_agents = '7 Agent Team'
elif '8_agent' in cwd:
    num_agents = '8 Agent Team'
else:
    raise Exception

og_num_agents = None
if 'zero_shot' in cwd:
    if '3_policy' in cwd:
        og_num_agents = '3 Agent Policy'
    elif '4_policy' in cwd:
        og_num_agents = '4 Agent Policy'
    elif '5_policy' in cwd:
        og_num_agents = '5 Agent Policy'
    else:
        raise Exception

for dir in dirs:
    # dummy_exp = pd.read_csv(dir + '/DummyExplainer_numbers.csv')
    # dummy_exp['Method'] = 'DummyExplainer'
    # dummy_exp['Layer'] = dir
    if dir == 'no_attn_ent':
        print('NO ATTN ENT PERFORMANCE:')
    elif dir == 'attn_ent':
        print('ATTN ENT PERFORMANCE:')
    else:
        raise Exception
    
    try:
        perf = pd.read_csv(dir + '/performance_numbers.csv',
                        names=["EpisodeReward", "Makespan", "Success Rate", "No Collisions Rate", "Collision Rate", "New Success Rate"],
                        skiprows=[0])
        
        print('Episode Reward Mean: ', mean(perf['EpisodeReward']))
        print('Episode Reward StdDev: ', std(perf['EpisodeReward']))

        print('Makespan Mean: ', mean(perf['Makespan']))
        print('Makespan StdDev: ', std(perf['Makespan']))

        print('Success Rate Mean: ', mean(perf['Success Rate']))
        print('Success Rate StdDev: ', std(perf['Success Rate']))

        print('No Collision Rate Mean: ', mean(perf['No Collisions Rate']))
        print('No Collision Rate StdDev: ', std(perf['No Collisions Rate']))

        print('Collision Rate Mean: ', mean(perf['Collision Rate']))
        print('Collision Rate StdDev: ', std(perf['Collision Rate']))

        print('New Success Rate Mean: ', mean(perf['New Success Rate']))
        print('New Success Rate StdDev: ', std(perf['New Success Rate']))

        
    except:
        pass

    graphmask_exp = pd.read_csv(dir + '/GraphMaskExplainer_numbers.csv')
    graphmask_exp['Method'] = 'GraphMaskExplainer'
    graphmask_exp['Layer'] = dir

    gnn_exp = pd.read_csv(dir + '/GNNExplainer_numbers.csv')
    gnn_exp['Method'] = 'GNNExplainer'
    gnn_exp['Layer'] = dir

    attn_exp = pd.read_csv(dir + '/AttentionExplainer_numbers.csv')
    attn_exp['Method'] = 'AttentionExplainer'
    attn_exp['Layer'] = dir

    cdf = pd.concat([
        # dummy_exp, 
        graphmask_exp, gnn_exp, attn_exp])
    cdfs.append(cdf)

cdf = pd.concat(cdfs)
# mdf = pd.melt(cdf, id_vars=['Method'], var_name=['Metric'])
ax = sns.pointplot(x = 'Method', y='FidPlus', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1, errorbar="sd")
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Layer", data=mdf[mdf['Metric'] == 'FidPlus'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="FidPlus", hue="Layer", data=cdf, showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)

# from statannot import add_stat_annotation
# add_stat_annotation(ax, data=cdf, x="Method", y="FidPlus", hue="Layer", box_pairs=box_pairs,
#                     test='Mann-Whitney', loc='inside', verbose=0)

plt.setp(ax.collections, alpha=.7)
# Means = cdf.groupby('Method')['FidPlus'].mean()
# plt.scatter(x=range(len(Means)),y=Means,c="k")
# sns.pointplot(x = 'Method', y='FidPlus', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
# sns.pointplot(
#     data=cdf, x="Method", y="FidPlus", hue="Layer",
#     dodge=.4, linestyles="none", errorbar=None,
#     markers="_"
# )
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
ax.get_legend().remove()
plt.xlabel('Explanation Method')
plt.ylabel('Positive Fidelity')
plt.title(task + ' with ' + num_agents) if og_num_agents is None else plt.title(task + ' with ' + num_agents + ' using ' + og_num_agents)
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-PositiveFidelity')
plt.show()

ax = sns.pointplot(x = 'Method', y='FidMinus', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1, errorbar="sd")
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Layer", data=mdf[mdf['Metric'] == 'FidMinus'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="FidMinus", hue="Layer", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)

from statannot import add_stat_annotation
add_stat_annotation(ax, data=cdf, x="Method", y="FidMinus", hue="Layer", box_pairs=box_pairs,
                    test='Mann-Whitney-gt', loc='inside', verbose=0)

plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='FidMinus', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
ax.get_legend().remove()
plt.xlabel('Explanation Method')
plt.ylabel('Negative Fidelity')
plt.title(task + ' with ' + num_agents) if og_num_agents is None else plt.title(task + ' with ' + num_agents + ' using ' + og_num_agents)
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-NegativeFidelity')
plt.show()


ax = sns.pointplot(x = 'Method', y='DeltaFid', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1, errorbar="sd")
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Layer", data=mdf[mdf['Metric'] == 'DeltaFid'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="DeltaFid", hue="Layer", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)

# from statannot import add_stat_annotation
# add_stat_annotation(ax, data=cdf, x="Method", y="DeltaFid", hue="Layer", box_pairs=box_pairs,
#                     test='Mann-Whitney', loc='inside', verbose=0)

plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='DeltaFid', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
ax.get_legend().remove()
plt.xlabel('Explanation Method')
plt.ylabel('Delta Fidelity')
plt.title(task + ' with ' + num_agents) if og_num_agents is None else plt.title(task + ' with ' + num_agents + ' using ' + og_num_agents)
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-DeltaFidelity')
plt.show()


ax = sns.pointplot(x = 'Method', y='Unfaithfulness', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1, errorbar="sd")
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
ax = sns.violinplot(x="Method", y="Unfaithfulness", hue="Layer", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)

# from statannot import add_stat_annotation
# add_stat_annotation(ax, data=cdf, x="Method", y="Unfaithfulness", hue="Layer", box_pairs=box_pairs,
#                     test='Mann-Whitney', loc='inside', verbose=0)


plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='Unfaithfulness', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
ax.get_legend().remove()
plt.xlabel('Explanation Method')
plt.ylabel('Unfaithfulness')
plt.title(task + ' with ' + num_agents) if og_num_agents is None else plt.title(task + ' with ' + num_agents + ' using ' + og_num_agents)
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-Unfaithfulness')
plt.show()


ax = sns.pointplot(x = 'Method', y='SymmetricUnfaithfulness', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1, errorbar="sd")
for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
    ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
ax = sns.violinplot(x="Method", y="SymmetricUnfaithfulness", hue="Layer", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)

# from statannot import add_stat_annotation
# add_stat_annotation(ax, data=cdf, x="Method", y="SymmetricUnfaithfulness", hue="Layer", box_pairs=box_pairs,
#                     test='Mann-Whitney', loc='inside', verbose=0)


plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='SymmetricUnfaithfulness', hue="Layer", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
ax.get_legend().remove()
plt.xlabel('Explanation Method')
plt.ylabel('SymmetricUnfaithfulness')
plt.title(task + ' with ' + num_agents) if og_num_agents is None else plt.title(task + ' with ' + num_agents + ' using ' + og_num_agents)
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-SymmetricUnfaithfulness')
plt.show()