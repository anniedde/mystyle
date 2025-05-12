import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

import torch
import os
import argparse
import json
import numpy as np

font_path = 'fonts/LinLibertine_R.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

device=torch.device('cuda')

def subplot(ax, map, metric, models, labels, arrow='', eval='EVALUATION TYPE'):
    """
    format of each map should be like this:
    lpips_map = {
        'lower_bound' : {
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        },
        'random' : { 
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        }
    }
    """
    print('models:', models)
    print('labels:', labels)
    cmap = plt.cm.get_cmap('rainbow')
    colors = [(r/255, g/255, b/255) for r, g, b in [(238, 102,119), (34,136,51),(242, 140, 40), (68,119,170)]]
    for i, model in enumerate(models):
        means = map[model]['mean']
        stds = map[model]['std']
        ax.plot(range(10), means, '-o', label=labels[i], color=colors[i])
        ax.fill_between(range(10), means - stds, means + stds, alpha=0.2, color=colors[i])
    ax.set_xticks(range(10), range(10))
    ax.set(xlabel='time', ylabel=metric)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_title(f'{eval} {metric}' + arrow)
    ax.title.set_size(24)
    #ax.legend()

def graph_reconstruction_results(celebs, model_list):
    lpips_map, psnr_map, dists_map, id_sim_map, fid_map, synth_id_sim_map = {}, {}, {}, {}, {}, {}
    """
    format of each map should be like this:
    lpips_map = {
        'lower_bound' : {
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        },
        'random' : { 
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        }
    }
    """
    for model in model_list:
        lpips_lists, psnr_lists, dists_lists, id_sim_lists, fid_lists, synth_id_sim_lists = [], [], [], [], [], []
        # for each celebrity, append a list of their metrics across time to each of the lists
        for celeb in celebs:
            lpips_list, psnr_list, dists_list, id_sim_list, fid_list, synth_id_sim_list = [], [], [], [], [], []
            for t in range(10):
                # get average lpips, psnr, dists, id_error for this model and test cluster
                if model == 'upper_bound':
                    reconstructions_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out',
                                                        celeb, 'reconstructions', model, str(t))
                else:
                    reconstructions_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out',
                                                        celeb, 'reconstructions', model, '9', str(t))
                metrics_path = os.path.join(reconstructions_folder, 'metrics.json')
                assert os.path.exists(metrics_path), f'{metrics_path} does not exist'
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                lpips_list.append(metrics['lpips'])
                psnr_list.append(metrics['psnr'])
                dists_list.append(metrics['dists'])
                if 'id_sim' not in metrics:
                    id_sim = 1 - metrics['id_error']
                else:
                    id_sim = metrics['id_sim']
                id_sim_list.append(id_sim)

                if model == 'upper_bound':
                    synth_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out',
                                                celeb, 'synthesis', model, 'all', str(t))
                else:
                    synth_folder = os.path.join('/playpen-nas-ssd/awang/mystyle_original/out',
                                                celeb, 'synthesis', model, '9', str(t))
                metrics_path = os.path.join(synth_folder, 'metrics.json')
                assert os.path.exists(metrics_path), f'{metrics_path} does not exist'
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                fid_list.append(metrics['fid'])
                if 'id_sim' not in metrics:
                    id_sim = 1 - metrics['id_error']
                else:
                    id_sim = metrics['id_sim']
                synth_id_sim_list.append(id_sim)

            lpips_lists.append(lpips_list)
            psnr_lists.append(psnr_list)
            dists_lists.append(dists_list)
            id_sim_lists.append(id_sim_list)
            fid_lists.append(fid_list)
            synth_id_sim_lists.append(synth_id_sim_list)
        
        # calculate mean and std for each metric
        lpips_map[model] = {
            'mean' : np.mean(lpips_lists, axis=0),
            'std' : np.std(lpips_lists, axis=0)
        }
        psnr_map[model] = {
            'mean' : np.mean(psnr_lists, axis=0),
            'std' : np.std(psnr_lists, axis=0)
        }
        dists_map[model] = {
            'mean' : np.mean(dists_lists, axis=0),
            'std' : np.std(dists_lists, axis=0)
        }
        id_sim_map[model] = {
            'mean' : np.mean(id_sim_lists, axis=0),
            'std' : np.std(id_sim_lists, axis=0)
        }
        fid_map[model] = {
            'mean' : np.mean(fid_lists, axis=0),
            'std' : np.std(fid_lists, axis=0)
        }
        synth_id_sim_map[model] = {
            'mean' : np.mean(synth_id_sim_lists, axis=0),
            'std' : np.std(synth_id_sim_lists, axis=0)
        }

    labels = ['Lower bound', 'ER-Rand', 'ER-Hull', 'Upper bound']
    #fig, axs = plt.subplots(3, 2)
    #tight layout
    #fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    #fig.set_figwidth(9)
    #fig.set_figheight(9)
    params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
    plt.rcParams.update(params)

    fig, axs = plt.subplots(2, 2)
    #tight layout
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    fig.set_figwidth(13)
    fig.set_figheight(10)
    #increase distance between subplots but not between subplots and the figure
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Increase font size
    plt.rcParams.update({'font.size': 20})
    # Increase font size of tick labels
    plt.rcParams.update({'xtick.labelsize': 20, 'ytick.labelsize': 20})

    subplot(axs[0, 0], lpips_map, 'LPIPS', models, labels, arrow=r'$\downarrow$', eval='Reconstruction')
    subplot(axs[0, 1], id_sim_map, 'ID Similarity', models, labels, arrow=r'$\uparrow$', eval='Reconstruction')
    subplot(axs[1, 0], fid_map, 'FID', models, labels, arrow=r'$\downarrow$', eval='Synthesis')
    subplot(axs[1, 1], synth_id_sim_map, 'ID Similarity', models, labels, arrow=r'$\uparrow$', eval='Synthesis')

    colors = [(r/255, g/255, b/255) for r, g, b in [(238, 102,119), (34,136,51),(242, 140, 40), (68,119,170)]]
    legend_elements = [Line2D([0], [0], marker='o', color=color, label=label, markerfacecolor=color, markersize=10, linestyle='-') 
                    for label, color in zip(labels, colors)]

    # Adjust the right margin of the subplots to create space for the legend
    plt.subplots_adjust(right=0.9)

    # Add the legend to the figure with increased font size
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path, size=20)
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.6), title='Models', fontsize='xx-large', title_fontsize=20, framealpha=1.0 ,prop=prop)

    #fig.suptitle('2D Evaluation After Training On All Clusters')
    plt.savefig(os.path.join('/playpen-nas-ssd/awang/mystyle_original/vis/fig3.png'), dpi=500)

    plt.clf()

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = ['lower_bound', 'constrained_random_3', 'constrained_ransac_3', 'upper_bound']
    celebs = ['Harry', 'Margot', 'IU', 'Michael', 'Sundar']
    graph_reconstruction_results(celebs, models)