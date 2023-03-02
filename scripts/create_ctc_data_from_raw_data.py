from pathlib import Path
import cv2
import numpy as np
import re
from skimage.measure import label
from tqdm import tqdm

moma = True

if moma:
    dataset = 'moma'
else:
    dataset = '2D'

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data') / dataset / 'test'

raw_data_path = datapath / 'raw_data'
raw_data_fps = sorted(list((raw_data_path / 'inputs').glob('*.png')))

ctc_path = datapath / 'ctc_data'

old_framenb = np.inf
meta_data = None
dataset_counter = 0
target_size = (256,32)

for idx,raw_data_fp in enumerate(tqdm(raw_data_fps)):

    framenb = int(re.findall('\d+',raw_data_fp.name)[-1])

    inputs = cv2.imread(str(raw_data_path / 'inputs' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16) 
    outputs = cv2.imread(str(raw_data_path / 'outputs' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    img = cv2.imread(str(raw_data_path / 'img' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    previmg = cv2.imread(str(raw_data_path / 'previmg' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)

    if framenb < old_framenb:
        if idx != 0:
            np.savetxt(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / ('man_track.txt'),meta_data,fmt='%d')
        dataset_counter += 1
        (ctc_path / f'{dataset_counter:02d}').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT' / 'TRA').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT' / 'SEG').mkdir(exist_ok=True)
        
        reset=True
        img_counter = -1
    else:
        reset=False

    old_framenb = framenb

    inputs_cellnbs = np.unique(inputs)
    inputs_cellnbs = inputs_cellnbs[inputs_cellnbs != 0]

    if reset:

        for inputs_cellnb in inputs_cellnbs:
            new_cell = np.array([inputs_cellnb,framenb-1,framenb-1,0],dtype=np.uint32)[None]
            if reset:
                meta_data = new_cell
                reset = False
            else:
                meta_data = np.concatenate((meta_data,new_cell),axis=0)
        img_counter += 1
        cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}' / f't{img_counter:03d}.tif'),previmg)
        cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / f'man_track{img_counter:03d}.tif'),inputs)
        cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'SEG' / f'man_seg{img_counter:03d}.tif'),inputs)
    else:
        track_gt_cellnbs = np.unique(track_gt)
        track_gt_cellnbs = track_gt_cellnbs[track_gt_cellnbs != 0]

        outputs_cellnbs = np.unique(outputs)
        outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != 0]

        for outputs_cellnb in outputs_cellnbs:
            if outputs_cellnb not in inputs_cellnbs:
                outputs[outputs == outputs_cellnb] = max(np.max(inputs),np.max(outputs)) + 1

        inputs_copy = np.copy(inputs)
        outputs_copy = np.copy(outputs)

        for track_gt_cellnb in track_gt_cellnbs:
            inputs_cellnb = inputs_copy[track_gt==track_gt_cellnb][0]
            assert inputs_cellnb != 0, f'Inconsistent GT between previous and current frame for ROI {dataset_counter-1} Frame {framenb}'
            inputs[inputs_copy == inputs_cellnb] = track_gt_cellnb
            outputs[outputs_copy == inputs_cellnb] = track_gt_cellnb

    max_cellnb = meta_data.shape[0]
    track_gt = np.copy(outputs).astype(np.uint16)

    outputs_cellnbs = np.unique(outputs)
    outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != 0]

    for outputs_cellnb in outputs_cellnbs:
        mask = outputs == outputs_cellnb
        mask_label = label(mask)
        mask_cellnbs = np.unique(mask_label)
        mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

        # for mask_cellnb in mask_cellnbs:
        #     if np.sum(mask[mask_label == mask_cellnb]) < 5:
        #         outputs[mask_label == mask_cellnb] = 0

        mask = outputs == outputs_cellnb
        mask_label = label(mask)
        mask_cellnbs = np.unique(mask_label)
        mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

        if len(mask_cellnbs) == 2:
            max_cellnb += 1
            track_gt[mask_label == mask_cellnbs[0]] = max_cellnb
            div_cell_1 = np.array([max_cellnb,framenb,framenb,outputs_cellnb],dtype=np.uint32)[None]
            max_cellnb += 1
            track_gt[mask_label == mask_cellnbs[1]] = max_cellnb
            div_cell_2 = np.array([max_cellnb,framenb,framenb,outputs_cellnb],dtype=np.uint32)[None]
            meta_data = np.concatenate((meta_data,div_cell_1,div_cell_2),axis=0)
        elif len(mask_cellnbs) > 2:
            raise NotImplementedError

        else:
            if outputs_cellnb in np.unique(inputs):
                meta_data[outputs_cellnb-1,2] = framenb 
            else:
                max_cellnb += 1
                new_cell = np.array([max_cellnb,framenb,framenb,outputs_cellnb],dtype=np.uint32)[None]
                meta_data = np.concatenate((meta_data,new_cell),axis=0)

    img_counter += 1
    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}' / f't{img_counter:03d}.tif'),img)
    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / f'man_track{img_counter:03d}.tif'),track_gt)
    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'SEG' / f'man_seg{img_counter:03d}.tif'),track_gt)

np.savetxt(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / ('man_track.txt'),meta_data,fmt='%d')