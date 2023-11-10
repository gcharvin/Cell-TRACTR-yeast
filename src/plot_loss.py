#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import re

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results')
folder = '230809_2D_CoMOT_track_two_stage_dn_track_dn_track_group_dab_intermediate_mask_3_enc_3_dec_layers'
folder = '231023_moma_flex_div_CoMOT_track_two_stage_dn_track_dab_intermediate_mask_4_enc_4_dec_layers'

with open(datapath / folder / 'metrics_train.pkl', 'rb') as f:
    metrics_train = pickle.load(f)

with open(datapath / folder / 'metrics_val.pkl', 'rb') as f:
    metrics_val = pickle.load(f)

losses = [key for key in metrics_train.keys() if 'loss' in key and not bool(re.search('\d',key)) and not np.isnan(metrics_train[key]).all()]
metrics = [key for key in metrics_train.keys() if 'acc' in key]
epochs = metrics_train['loss'].shape[0]
epochs_val = metrics_val['loss'].shape[0]


groups = [None]

training_methods = [loss[8:] for loss in losses if 'loss_ce_' in loss]

for training_method in training_methods:
    if 'loss_ce_' + training_method in losses:
        groups += [training_method]

fig,ax = plt.subplots()
lrs = metrics_train['lr']

for i in range(lrs.shape[1]):
    ax.plot(np.arange(1,epochs+1),lrs[:,i])
ax.set_title('Learning Rate Schedule')
ax.set_xlabel('Epochs')
ax.set_ylabel('lr')
fig.savefig(datapath / folder / 'learning_rate.png')

# Plot Overall Loss
fig,ax = plt.subplots()

ax.plot(np.arange(1,epochs+1),np.nanmean(metrics_train['loss'],axis=-1),label='train')
ax.plot(np.arange(1,epochs_val+1),np.nanmean(metrics_val['loss'],axis=-1),label='val')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig(datapath / folder / 'loss_plot_overall.png')

ax.set_yscale('log')
plt.savefig(datapath / folder / 'loss_plot_overall_log.png')

fig,ax = plt.subplots(len(groups),2,figsize=(10,15))

min_y = np.inf
max_y = 0
for loss in losses:
    if loss == 'loss' or 'CoMOT' in loss or bool(re.search('\d',loss)) or np.isnan(metrics_train[loss]).all():
        continue
    category = [bool(re.search(g + '$',loss)) and (('dn' in g and 'dn' in loss) or 'dn' not in loss) for g in groups[1:]]

    if sum(category) == 0:
        g = 0
    else:
        g = category.index(True) + 1

    plot_epochs_train = np.arange(1,epochs+1)
    plot_epochs_val = np.arange(1,epochs+1)
    plot_epochs_train = plot_epochs_train[~np.isnan(metrics_train[loss]).all(-1)]
    plot_epochs_val = plot_epochs_val[~np.isnan(metrics_val[loss]).all(-1)]
    plot_metrics_train = metrics_train[loss][~np.isnan(metrics_train[loss]).all(-1)]
    plot_metrics_val = metrics_val[loss][~np.isnan(metrics_val[loss]).all(-1)]

    train_loss = np.nanmean(plot_metrics_train,axis=-1)
    val_loss = np.nanmean(plot_metrics_val,axis=-1)
    ax[g,0].plot(plot_epochs_train,train_loss,label=loss)
    ax[g,1].plot(plot_epochs_val,val_loss,label=loss)
    min_y = min((min_y,min(train_loss),min(val_loss)))
    max_y = max((max_y,max(train_loss),max(val_loss)))


for g,group in enumerate(groups):
    for i, dataset in enumerate(['train','val']):
        ax[g,i].set_xlabel('Epochs')
        ax[g,i].set_ylabel('Loss')
        ax[g,i].legend()
        ax[g,i].set_title(f'{"Overall" if group is None else group}: {dataset}')
        ax[g,i].set_ylim(min_y,max_y)
        ax[g,i].set_xlim(1,epochs)

fig.tight_layout()
plt.savefig(datapath / folder / 'loss_plot.png')

for g in range(len(groups)):
    ax[g,0].set_yscale('log')
    ax[g,1].set_yscale('log')

plt.savefig(datapath / folder / 'loss_plot_log.png')

    
# losses = [loss for loss in losses if 'mask' not in loss and 'dice' not in loss and loss not in ['loss','loss_ce_enc','loss_bbox_enc','loss_giou_enc']] # Auxillary losses does not have mask / dice loss
losses = [loss for loss in losses if loss not in ['loss','loss_ce_enc','loss_bbox_enc','loss_giou_enc']] # Auxillary losses does not have mask / dice loss

if 'loss_mask_1' in metrics_train and np.isnan(metrics_train['loss_mask_1']).all():
    losses = [loss for loss in losses if 'mask' not in loss and 'dice' not in loss]

if 'enc' in groups:
    groups.remove('enc')

if 'loss_ce_0_CoMOT' in metrics_train.keys():
    losses += ['loss_ce_CoMOT','loss_bbox_CoMOT','loss_giou_CoMOT']
    if 'loss_mask' in losses:
        losses += ['loss_mask_CoMOT','loss_dice_CoMOT']
    groups += ['CoMOT']

def plot_aux_losses(losses,metrics_train,metrics_val,groups):

    for gidx,group in enumerate(groups):
        other_groups = [g for g in groups if g != group]

        if gidx == 0:
            losses_group = [loss for loss in losses if sum([g in loss for g in other_groups]) == 0]
        else:
            losses_group = [loss for loss in losses if group in loss and not ('group' in loss and 'group' not in group)]

        fig,ax = plt.subplots(len(losses_group),2,figsize=(10,len(losses_group)*3))
        min_y = np.inf
        max_y = 0
        for i,loss in enumerate(losses_group):
            if group is not None:
                loss = loss.replace('_' + group,'')

            if group == 'CoMOT' and loss == 'loss_ce' and np.isnan(metrics_train['loss_ce_0_CoMOT']).all():
                continue
                
            for loss_key in metrics_train.keys():
                # if (group is not None and group not in loss_key and bool(re.search('\D$',loss_key)) and not loss_key==loss) or (group is None and bool(re.search('\D$',loss_key)) and not loss_key==loss):
                #     continue

                if (group is None and loss in loss_key and (bool(re.search('\d$',loss_key)) or loss_key == loss)) or (group is not None and group in loss_key and loss in loss_key and not ('group' in loss_key and 'group' not in group)):

                    if loss in loss_key:# and bool(re.search('\d',loss_key)):
                        plot_epochs_train = np.arange(1,epochs+1)
                        plot_epochs_val = np.arange(1,epochs+1)
                        plot_epochs_train = plot_epochs_train[~np.isnan(metrics_train[loss_key]).all(-1)]
                        plot_epochs_val = plot_epochs_val[~np.isnan(metrics_val[loss_key]).all(-1)]
                        plot_metrics_train = metrics_train[loss_key][~np.isnan(metrics_train[loss_key]).all(-1)]
                        plot_metrics_val = metrics_val[loss_key][~np.isnan(metrics_val[loss_key]).all(-1)]
                        train_loss = np.nanmean(plot_metrics_train,axis=-1)
                        val_loss = np.nanmean(plot_metrics_val,axis=-1)
                        ax[i,0].plot(plot_epochs_train,train_loss,label=loss_key)
                        ax[i,1].plot(plot_epochs_val,val_loss,label=loss_key)
                        min_y = min((min_y,np.nanmin(train_loss),np.nanmin(val_loss)))
                        max_y = max((max_y,np.nanmax(train_loss),np.nanmax(val_loss)))

            for d,dataset in enumerate(['train','val']):
                ax[i,d].set_xlabel('Epochs')
                ax[i,d].set_ylabel('Loss')
                ax[i,d].legend()
                ax[i,d].set_title(f'{dataset}: {loss} {group if group is not None else ""}')
                ax[i,d].set_ylim(min_y,max_y)
                ax[i,d].set_xlim(1,epochs)

        fig.tight_layout()
        plt.savefig(datapath / folder / (f'aux_loss{"_" + group if group is not None else ""}_plot.png'))

plot_aux_losses(losses,metrics_train,metrics_val,groups=groups)
metrics.remove('post_division_track_acc')

if 'object_detection_only' in folder:
    metrics.remove('overall_track_acc')
    metrics.remove('divisions_track_acc')
    metrics.remove('new_cells_track_acc')

# Plot acc
fig,ax = plt.subplots(1,2,figsize=(10,5))
colors = ['b','g','r','c','m','y']
metrics_txt = []
# index = metrics_val['loss'].mean(-1).argmin()
# metrics_txt.append(f'Best epoch: {index+1}\n')
print(folder)
for midx,metric in enumerate(metrics):

    i = 0 if 'det_acc' in metric else 1
    
    train_acc = np.nansum(metrics_train[metric],axis=-2)
    train_acc = train_acc[:,0] / train_acc[:,1]
    val_acc = np.nanmean(metrics_val[metric],axis=-2)
    val_acc = val_acc[:,0] / val_acc[:,1]

    replace_word = '_det_acc' if i ==0 else '_track_acc'
    metric = metric.replace(replace_word,'')

    ax[i].plot(np.arange(1,epochs+1),train_acc, color = colors[midx] if 'overall' not in metric else 'k',label=metric)
    ax[i].plot(np.arange(1,epochs_val+1),val_acc, '--', color = colors[midx] if 'overall' not in metric else 'k',)

    if metric in ['overall', 'mask', 'bbox']:
        metric = 'track' if metric == 'overall' else metric
        print(f'{metric}\nTrain: {train_acc[-1]}\nVal: {val_acc[-1]}')
        # metrics_txt.append(f'{metric}\nTrain: {train_acc[index]}\nVal: {val_acc[index]}\n')
        metrics_txt.append(f'{metric}\nTrain: {train_acc[-1]}\nVal: {val_acc[-1]}\n')
    
with open(str(datapath / folder / 'metrics.txt'),'w') as f:
    for txt in metrics_txt:
        f.write(txt)

for i in range(2):
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Accuracy')
    ax[i].legend()
    ax[i].set_ylim(0,1)

ax[0].set_title('Object Detection Accuracy')
ax[1].set_title('Tracking Accuracy')

plt.savefig(datapath / folder / 'acc_plot.png')
