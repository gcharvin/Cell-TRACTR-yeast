#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import re

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results')
folder = '20221204_fix_bug_no_dab_no_mask'
# folder = '20221203_dab_no_mask'

with open(datapath / folder / 'metrics_train.pkl', 'rb') as f:
    metrics_train = pickle.load(f)

with open(datapath / folder / 'metrics_val.pkl', 'rb') as f:
    metrics_val = pickle.load(f)

losses = [key for key in metrics_train.keys() if 'loss' in key and not bool(re.search('\d',key))]
metrics = [key for key in metrics_train.keys() if 'acc' in key]
epochs = metrics_train['loss'].shape[0]
epochs_val = metrics_train['loss'].shape[0]


# Plot Overall Loss
fig,ax = plt.subplots()

ax.plot(np.arange(1,epochs+1),np.nanmean(metrics_train['loss'],axis=-1),label='train')
ax.plot(np.arange(1,epochs+1),np.nanmean(metrics_val['loss'],axis=-1),label='val')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig(datapath / folder / 'loss_plot_overall.png')

ax.set_yscale('log')
plt.savefig(datapath / folder / 'loss_plot_overall_log.png')



fig,ax = plt.subplots(1,2)

min_y = np.inf
max_y = 0
for loss in losses:
    if loss == 'loss' or bool(re.search('\d',loss)):
        continue
    train_loss = np.nanmean(metrics_train[loss],axis=-1)
    val_loss = np.nanmean(metrics_val[loss],axis=-1)
    ax[0].plot(np.arange(1,epochs+1),train_loss,label=loss)
    ax[1].plot(np.arange(1,epochs_val+1),val_loss,label=loss)
    min_y = min((min_y,min(train_loss),min(val_loss)))
    max_y = max((max_y,max(train_loss),max(val_loss)))

for i, dataset in enumerate(['train','val']):
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Loss')
    ax[i].legend()
    ax[i].set_title(dataset)
    ax[0].set_ylim(min_y,max_y)

fig.tight_layout()
plt.savefig(datapath / folder / 'loss_plot_train.png')

ax[0].set_yscale('log')
ax[1].set_yscale('log')

plt.savefig(datapath / folder / 'loss_plot_train_log.png')


losses = [loss for loss in losses if loss not in ['loss','loss_mask','loss_dice']] # total loss has not auxillary losses
fig,ax = plt.subplots(2,len(losses),figsize=(15,6))

min_y = np.inf
max_y = 0
for i,loss in enumerate(losses):
    for loss_key in metrics_train.keys():
        if loss in loss_key and loss != loss_key:
            train_loss = np.nanmean(metrics_train[loss_key],axis=-1)
            val_loss = np.nanmean(metrics_val[loss_key],axis=-1)
            ax[0,i].plot(np.arange(1,epochs+1),train_loss,label=loss_key)
            ax[1,i].plot(np.arange(1,epochs_val+1),val_loss,label=loss_key)
            min_y = min((min_y,min(train_loss),min(val_loss)))
            max_y = max((max_y,max(train_loss),max(val_loss)))

    for d,dataset in enumerate(['train','val']):
        ax[d,i].set_xlabel('Epochs')
        ax[d,i].set_ylabel('Loss')
        ax[d,i].legend()
        ax[d,i].set_title(f'{dataset}: {loss}')
        ax[d,i].set_ylim(min_y,max_y)

fig.tight_layout()
plt.savefig(datapath / folder / 'aux_loss_plot_train.png')


# Plot acc
fig,ax = plt.subplots()
colors = ['b','g','r','c']
for midx,metric in enumerate(metrics):
    train_acc = np.nanmean(metrics_train[metric],axis=-2)
    train_acc = train_acc[:,0] / train_acc[:,1]
    val_acc = np.nanmean(metrics_val[metric],axis=-2)
    val_acc = val_acc[:,0] / val_acc[:,1]
    ax.plot(np.arange(1,epochs+1),train_acc, color = colors[midx],label='train_' + metric)
    ax.plot(np.arange(1,epochs_val+1),val_acc, '--', color = colors[midx],label='val_' + metric)

ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
ax.set_ylim(0,1)

plt.savefig(datapath / folder / 'acc_plot.png')
