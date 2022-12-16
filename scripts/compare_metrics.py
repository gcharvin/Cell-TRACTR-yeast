

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import re

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results')
folders = list(datapath.iterdir())
filter = '221210_'
# folder_paths = [folder for folder in folders if filter in folder.name and 'dab' in folder.name and 'attn' not in folder.name]
# folder_paths = [folder for folder in folders if filter in folder.name and ('attn' in folder.name or 'dab' in folder.name)]
folder_paths = [folder for folder in folders if '221212_reduced_loss_coef_1_2_2_dab_no_mask' in folder.name or '221210_no_attn_mask_object_track_emb_dab_no_mask' in folder.name]
# folder_paths = [folder for folder in folders if '221210_dab_no_mask' in folder.name or '221211_dn_track_rand_embed_dab_no_mask' in folder.name or '221210_dn_object_dab_no_mask' in folder.name]

comparisons_folder = 'Comparisons'
save_folder = 'placeholder_folder_name'
(datapath / comparisons_folder).mkdir(exist_ok=True)
(datapath / comparisons_folder / save_folder).mkdir(exist_ok=True)
with open(folder_paths[0] / 'metrics_train.pkl', 'rb') as f:
        metrics_train = pickle.load(f)

replace_words = ['_object_det_acc', '_track_acc']

metrics_d_t = []
axes = []
figs = []
axes_train_val = []
figs_train_val = []
for replace_word in replace_words:
    metrics_d_t.append([key for key in metrics_train.keys() if replace_word in key])
    fig,ax = plt.subplots(1,len(metrics_d_t[-1]),figsize=(16,8))
    axes.append(ax)
    figs.append(fig)

    fig_train, ax_train = plt.subplots(1,len(metrics_d_t[-1]),figsize=(16,8))
    fig_val, ax_val = plt.subplots(1,len(metrics_d_t[-1]),figsize=(16,8))

    axes_train_val.append([ax_train,ax_val])
    figs_train_val.append([fig_train,fig_val])

suptitles = ['Obect Detection Accuracy','Tracking Accuracy']

for fidx,folder_path in enumerate(folder_paths):

    with open(folder_path / 'metrics_train.pkl', 'rb') as f:
        metrics_train = pickle.load(f)

    with open(folder_path / 'metrics_val.pkl', 'rb') as f:
        metrics_val = pickle.load(f)

    epochs = metrics_train['loss'].shape[0]
    epochs_val = metrics_val['loss'].shape[0]

    colors = ['b','g','r','c','m','y']

    label = folder_path.name[7:] #trivial way to get rid of date
    label = folder_path.name.replace('_no_mask','')

    for i,metrics in enumerate(metrics_d_t):
        ax = axes[i]
        replace_word = replace_words[i]
        for midx,metric in enumerate(metrics):
                        
            train_acc = np.nanmean(metrics_train[metric],axis=-2)
            train_acc = train_acc[:,0] / train_acc[:,1]
            val_acc = np.nanmean(metrics_val[metric],axis=-2)
            val_acc = val_acc[:,0] / val_acc[:,1]

            
            ax[midx].plot(np.arange(1,epochs+1),train_acc*100, color = colors[fidx] ,label=label)
            ax[midx].plot(np.arange(1,epochs_val+1),val_acc*100, '--', color = colors[fidx])

            axes_train_val[i][0][midx].plot(np.arange(1,epochs+1),train_acc*100, color = colors[fidx] ,label=label)
            axes_train_val[i][1][midx].plot(np.arange(1,epochs_val+1),val_acc*100, color = colors[fidx] ,label=label)

for i,metrics in enumerate(metrics_d_t):
    ax = axes[i]
    fig = figs[i]
    fig.suptitle(suptitles[i],fontsize=20)
    for m in range(len(metrics)):
        ax[m].set_xlabel('Epochs',fontsize=15)
        ax[m].set_ylim(0,100)
        ax[m].set_title(metrics[m].replace(replace_words[i],''),fontsize=15)

        for j in range(2):
            axes_train_val[i][j][m].set_xlabel('Epochs',fontsize=15)
            axes_train_val[i][j][m].set_ylim(0,100)
            axes_train_val[i][j][m].set_title(metrics[m].replace(replace_words[i],''),fontsize=15)

    ax[0].set_ylabel('Accuracy',fontsize=15)
    ax[0].legend(fontsize=10)

    for j in range(2):
        axes_train_val[i][j][0].set_ylabel('Accuracy',fontsize=15)
        axes_train_val[i][j][0].legend(fontsize=10)
        word = 'train' if j == 0 else 'validation'
        figs_train_val[i][j].suptitle(word.capitalize() + ': ' + suptitles[i],fontsize=20)
        figs_train_val[i][j].savefig(folder_path.parent / comparisons_folder / save_folder / (f'metrics{replace_words[i]}_{word}.png'))
        

    fig.savefig(folder_path.parent / comparisons_folder / save_folder / (f'metrics{replace_words[i]}.png'))