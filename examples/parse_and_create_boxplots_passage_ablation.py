import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean, std
import os

dirs = ['15x', '20x', '30x', '50x']
cdfs = []

for dir in dirs:
    # dummy_exp = pd.read_csv(dir + '/DummyExplainer_numbers.csv')
    # dummy_exp['Method'] = 'DummyExplainer'
    # dummy_exp['Attention Entropy Weight'] = dir

    try:
        print(dir + ' PERFORMANCE:')
        perf = pd.read_csv(dir + '/performance_numbers.csv',
                        names=["EpisodeReward", "Makespan", "Success Rate", "No Collisions Rate", "No Obstacle Collisions Rate"],
                        skiprows=[0])
        
        print('Episode Reward Mean: ', mean(perf['EpisodeReward']))
        print('Episode Reward StdDev: ', std(perf['EpisodeReward']))

        print('Makespan Mean: ', mean(perf['Makespan']))
        print('Makespan StdDev: ', std(perf['Makespan']))

        print('Success Rate Mean: ', mean(perf['Success Rate']))
        print('Success Rate StdDev: ', std(perf['Success Rate']))

        print('No Collision Rate Mean: ', mean(perf['No Collisions Rate']))
        print('No Collision Rate StdDev: ', std(perf['No Collisions Rate']))

        print('No Obs Coll Rate Mean: ', mean(perf['No Obstacle Collisions Rate']))
        print('No Obs Coll Rate StdDev: ', std(perf['No Obstacle Collisions Rate']))

        
    except:
        pass

    graphmask_exp = pd.read_csv(dir + '/GraphMaskExplainer_numbers.csv')
    graphmask_exp['Method'] = 'GraphMaskExplainer'
    graphmask_exp['Attention Entropy Weight'] = dir

    gnn_exp = pd.read_csv(dir + '/GNNExplainer_numbers.csv')
    gnn_exp['Method'] = 'GNNExplainer'
    gnn_exp['Attention Entropy Weight'] = dir

    attn_exp = pd.read_csv(dir + '/AttentionExplainer_numbers.csv')
    attn_exp['Method'] = 'AttentionExplainer'
    attn_exp['Attention Entropy Weight'] = dir

    cdf = pd.concat([
        # dummy_exp, 
        graphmask_exp, gnn_exp, attn_exp])
    cdfs.append(cdf)

cdf = pd.concat(cdfs)
# mdf = pd.melt(cdf, id_vars=['Method'], var_name=['Metric'])
ax = sns.pointplot(x = 'Method', y='FidPlus', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.6, capsize=.1, errorbar="sd")
# for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
#     ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Attention Entropy Weight", data=mdf[mdf['Metric'] == 'FidPlus'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="FidPlus", hue="Attention Entropy Weight", data=cdf, showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)
plt.setp(ax.collections, alpha=.7)
# Means = cdf.groupby('Method')['FidPlus'].mean()
# plt.scatter(x=range(len(Means)),y=Means,c="k")
# sns.pointplot(x = 'Method', y='FidPlus', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
# sns.pointplot(
#     data=cdf, x="Method", y="FidPlus", hue="Attention Entropy Weight",
#     dodge=.4, linestyles="none", errorbar=None,
#     markers="_"
# )
leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
plt.xlabel('Explanation Method')
plt.ylabel('Positive Fidelity')
plt.title('Positive Fidelity')
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-PositiveFidelity')
plt.show()

ax = sns.pointplot(x = 'Method', y='FidMinus', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.6, capsize=.1, errorbar="sd")
# for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
#     ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Attention Entropy Weight", data=mdf[mdf['Metric'] == 'FidMinus'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="FidMinus", hue="Attention Entropy Weight", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)
plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='FidMinus', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
plt.xlabel('Explanation Method')
plt.ylabel('Negative Fidelity')
plt.title('Negative Fidelity')
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-NegativeFidelity')
plt.show()


ax = sns.pointplot(x = 'Method', y='DeltaFid', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.6, capsize=.1, errorbar="sd")
# for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
#     ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
# ax = sns.boxplot(x="Method", y="value", hue="Attention Entropy Weight", data=mdf[mdf['Metric'] == 'DeltaFid'],showmeans=True, meanline=True)    
ax = sns.violinplot(x="Method", y="DeltaFid", hue="Attention Entropy Weight", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)
plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='DeltaFid', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
plt.xlabel('Explanation Method')
plt.ylabel('Delta Fidelity')
plt.title('Delta Fidelity')
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-DeltaFidelity')
plt.show()


ax = sns.pointplot(x = 'Method', y='Unfaithfulness', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.6, capsize=.1, errorbar="sd")
# for (x0, y0), (x1, y1) in zip(ax.collections[0].get_offsets(), ax.collections[1].get_offsets()):
#     ax.plot([x0, x1], [y0, y1], color='black', zorder=0)
ax = sns.violinplot(x="Method", y="Unfaithfulness", hue="Attention Entropy Weight", data=cdf,showmeans=True, meanline=True, inner=None, cut=0, linewidth=0)
plt.setp(ax.collections, alpha=.7)
# sns.pointplot(x = 'Method', y='Unfaithfulness', hue="Attention Entropy Weight", data=cdf, estimator=mean, join=False, dodge=.4, capsize=.1)
leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, ['No Attention Entropy', 'Attention Entropy', 'No Attention Entropy $\mu \pm \sigma$', 'Attention Entropy $\mu \pm \sigma$'], title='')
plt.xlabel('Explanation Method')
plt.ylabel('Unfaithfulness')
plt.title('Unfaithfulness')
# plt.show()
plt.savefig(os.getcwd()[37:].replace('/', '-') + '-Unfaithfulness')
plt.show()