# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:26:29 2022

@author: 17742
"""

from pathlib import Path
import cv2
import numpy as np 
from skimage.measure import label
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_avg_cell_area_norm(fps,num_samples = 30):
    area = []
    fps_rand = np.random.choice(fps,num_samples)
    
    for fp in fps_rand:
        inputs = cv2.imread(str(datapath / 'inputs' / fp.name),cv2.IMREAD_ANYDEPTH)    
        inputs_cells = np.unique(inputs)[1:]
        
        for inputs_cell in inputs_cells:
            a = np.sum(inputs == inputs_cell) / (inputs.shape[0] * inputs.shape[1])
            area.append(a)
        
        outputs = cv2.imread(str(datapath / 'outputs' / fp.name),cv2.IMREAD_ANYDEPTH) 
        seg_label = label(outputs>0)
        seg_cells = np.unique(seg_label)[1:]
        
        for seg_cell in seg_cells:
            a = np.sum(seg_label == seg_cell) / (outputs.shape[0] * outputs.shape[1])
            area.append(a)
        
    return sum(area) / len(area)
        
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/new_dataset/raw_data')

fps = sorted(list((datapath / 'img').glob("*.png")))[:]
# fps = [fp for fp in fps if 'TrainingSet' not in fp.stem]

np.random.seed(1)
colors = tuple([(255*np.random.random(3)).astype(np.uint8) for _ in range(12)])

save_dummy = True
if save_dummy:
    (datapath / 'dummy').mkdir(exist_ok=True)
remove = True
resize = 4
avg_area_norm = get_avg_cell_area_norm(fps,num_samples=30)
correct = []

for fp in tqdm(fps):
    if fp.stem == '':
        print(fp.stem)
    inputs = cv2.imread(str(datapath / 'inputs' / fp.name),cv2.IMREAD_ANYDEPTH)    
    outputs = cv2.imread(str(datapath / 'outputs' / fp.name),cv2.IMREAD_ANYDEPTH)
    previmg = cv2.imread(str(datapath / 'previmg' / fp.name))
    img = cv2.imread(str(datapath / 'img' / fp.name))
    
    prevseg = (inputs > 0) * 1
    seg = (outputs > 0) * 1
    
    prevseg_label = label(prevseg.astype(np.uint8))
    prevseg_cells = np.unique(prevseg_label)[1:]
    
    seg_label = label(seg.astype(np.uint8))
    seg_cells = np.unique(seg_label)[1:]
    
    pred = np.zeros((inputs.shape))
    i = 0
    min_area = 50 / (inputs.shape[0] * inputs.shape[1])
    
    if len(prevseg_cells) == 0 or len(seg_cells) == 0:
        continue
    
    for idx,prevseg_cell in enumerate(prevseg_cells):  
        y_in,_ = np.where(prevseg_label == prevseg_cell)
        y_in = np.min(y_in)
        if i == len(seg_cells):
            continue
        prev_area = np.sum(prevseg_label == prevseg_cell) / (prevseg_label.shape[0] * prevseg_label.shape[1])
        
        if prev_area < min_area:
            continue
        
        for _ in range(len(seg_cells)):
            if i == len(seg_cells):
                break
            cur_area = np.sum(seg_label == seg_cells[i]) / (seg_label.shape[0] * seg_label.shape[1])
            area_ratio = cur_area / prev_area
            y_out,_ = np.where(seg_label==seg_cells[i])
            y_out = np.median(y_out)
            
            if y_out < y_in:
                i += 1
                continue
                        
            if cur_area < min_area:
                i += 1
                continue
                
            if i == len(seg_cells) - 1:
                pred[seg_label == seg_cells[i]] = prevseg_cell
                i += 1
                break
            
            if prev_area > 1.5*avg_area_norm and  area_ratio > 0.9 and area_ratio < 1.3:
               pred[seg_label == seg_cells[i]] = prevseg_cell
               i += 1
               break

            elif prev_area < 0.8*avg_area_norm and area_ratio > 0.6 and area_ratio < 1.75:
               pred[seg_label == seg_cells[i]] = prevseg_cell
               i += 1
               break
            
            elif prev_area < 1.5*avg_area_norm and area_ratio > 0.75 and area_ratio < 1.75:
               pred[seg_label == seg_cells[i]] = prevseg_cell
               i += 1
               break
               
            else:
                                  
                cur_area_div = np.sum(seg_label == seg_cells[i+1]) / (seg_label.shape[0] * seg_label.shape[1])
                
                area_div_ratio = (cur_area + cur_area_div) / prev_area
                
                if i == len(seg_cells) - 2 or (area_div_ratio > 0.8 and area_div_ratio < 1.5):
                    pred[seg_label == seg_cells[i]] = prevseg_cell
                    pred[seg_label == seg_cells[i+1]] = prevseg_cell
                    
                    i += 2
                break
            
            
        previmg[prevseg_label == prevseg_cell] = colors[idx]
        y,x = np.where(prevseg_label==prevseg_cell)
        y,x = int(np.median(y)), int(np.median(x))
        previmg = cv2.putText(previmg,str(prevseg_cell),
                        org=(np.max((0,x-3)),y+5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.6,
                        color = (0,0,0),
                        thickness = 1)
    
        img[pred == prevseg_cell] = colors[idx]
        
        blah_mask = label(pred == prevseg_cell)
        blah_cells = np.unique(blah_mask)[1:]
        
        for blah_cell in blah_cells:
            y,x = np.where(blah_mask==blah_cell)
            y,x = int(np.median(y)), int(np.median(x))
            img = cv2.putText(img,str(prevseg_cell),
                            org=(np.max((0,x-3)),y+5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.6,
                            color = (0,0,0),
                            thickness = 1)
            

    outputs_cells = np.unique(outputs)
    seg_cells = np.unique(seg_label)[1:]
    
    for seg_cell in seg_cells:
        
        pred_nb = pred[seg_label == seg_cell][0]
        output_nb = outputs[seg_label == seg_cell][0]
        
        if output_nb not in inputs or np.sum(seg_label==seg_cell) / (seg_label.shape[0]*seg_label.shape[1]) < min_area or np.sum(inputs==output_nb) / (seg_label.shape[0]*seg_label.shape[1]) < min_area:
            continue
        
        inputs_pred_nb = inputs[prevseg_label == pred_nb][0]
        
        if output_nb == inputs_pred_nb:
            correct.append(1)
            # print('right')
        else:
            correct.append(0)
            print(f'{fp.stem}:wrong')
            
            if save_dummy:
                cv2.imwrite(str(datapath / 'dummy' / fp.name),np.concatenate((previmg,img,np.repeat(prevseg[:,:,None]*255,3,axis=-1),np.repeat(seg[:,:,None]*255,3,axis=-1)),axis=1))
    
            plot = np.concatenate((previmg,img,np.repeat(prevseg[:,:,None]*255,3,axis=-1),np.repeat(seg[:,:,None]*255,3,axis=-1)),axis=1)

            # plt.show(plot)

            while True:
                answer = input('Do you want to remove this sample? (y/n)')

                if answer == 'y':
                    if (datapath / 'inputs' / fp.name).exists():
                        for folder in ['inputs','outputs','img','previmg']:
                            (datapath / folder / fp.name).unlink()
                        print('file exists and is deleted')
                    else:
                        print('File does not exist or file was already deleted')
                    break
                elif answer == 'n':
                    break
                else:
                    print('Please select y or n')

            # plt.close('all')

print(f'{sum(correct)} correct predictions out of {len(correct)}')
    