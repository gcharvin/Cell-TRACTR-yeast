from pathlib import Path
import cv2
import numpy as np
import re
from skimage.measure import label
from tqdm import tqdm

moma = False

if moma:
    dataset = 'moma'
else:
    dataset = '2D'

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data') / dataset #/ 'test'
datapath = Path('/projectnb/dunlop/ooconnor/vanvliet_data')
datapath = Path('/projectnb/dunlop/ooconnor/16bit')

# raw_data_path = datapath / 'vanvliet_tracking'
raw_data_path = datapath / 'trainingset_16bit_20220121'
raw_data_fps = sorted(list((raw_data_path / 'inputs').glob('*.png')))

ctc_path = datapath / 'CTC'
ctc_path.mkdir(exist_ok=True)

old_framenb = np.inf
meta_data = None
dataset_counter = dataset_counter_init = 44
target_size = (256,32)
old_dataset = 'blah'

dataset_dict = {}
fnb = 0
split = 0

for idx,raw_data_fp in enumerate(tqdm(raw_data_fps)):

    dataset_dict[raw_data_fp.stem] = 0

    framenb = re.findall('\d+',raw_data_fp.name)[-1]
    dataset = raw_data_fp.name[:raw_data_fp.name.index(framenb)]
    framenb = int(framenb)

    if framenb < old_framenb or framenb > old_framenb + 1 or dataset != old_dataset:
        ctc_set = [raw_data_fp.stem]
        value = 1
    else:
        ctc_set.append(raw_data_fp.stem)
        value += 1

    for fn in ctc_set:
        dataset_dict[fn] = value

    old_framenb = framenb
    old_dataset = dataset


old_dataset = 'blah'
old_framenb = np.inf

for idx,raw_data_fp in enumerate(tqdm(raw_data_fps)):

    framenb = re.findall('\d+',raw_data_fp.name)[-1]
    dataset = raw_data_fp.name[:raw_data_fp.name.index(framenb)]
    framenb = int(framenb)

    if dataset_dict[raw_data_fp.stem] < 10 or 'vanvliet' in raw_data_fp.name:
        continue

    # if 'vanvliet_trpL_151021-12_Frame' in raw_data_fp.name:
    #     dataset_counter = 46
    # else:
    #     continue

    inputs = cv2.imread(str(raw_data_path / 'inputs' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16) 
    outputs = cv2.imread(str(raw_data_path / 'outputs' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    img = cv2.imread(str(raw_data_path / 'img' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    previmg = cv2.imread(str(raw_data_path / 'previmg' / raw_data_fp.name),cv2.IMREAD_ANYDEPTH).astype(np.uint16)

    if framenb < old_framenb or framenb > old_framenb + 1 or dataset != old_dataset:

        if dataset_counter != dataset_counter_init:
            np.savetxt(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / ('man_track.txt'),meta_data,fmt='%d')
        dataset_counter += 1
        (ctc_path / f'{dataset_counter:02d}').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT' / 'TRA').mkdir(exist_ok=True)
        (ctc_path / f'{dataset_counter:02d}_GT' / 'SEG').mkdir(exist_ok=True)
        file = open(str(ctc_path / f'{dataset_counter:02d}' / f'{dataset}.txt'), "w")
        file.write(dataset)
        file.close()
            
        reset=True
        img_counter = 0
    else:
        reset=False

    old_framenb = framenb

    inputs_cellnbs = np.unique(inputs)
    inputs_cellnbs = inputs_cellnbs[inputs_cellnbs != 0]

    for inputs_cellnb in inputs_cellnbs:
        mask = inputs == inputs_cellnb
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours[0].size < 6:
            inputs[inputs == inputs_cellnb] == 0
            inputs_cellnbs = inputs_cellnbs[inputs_cellnbs != inputs_cellnb]

    outputs_cellnbs = np.unique(outputs)
    outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != 0]

    for outputs_cellnb in outputs_cellnbs:
        mask = outputs == outputs_cellnb
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_label = label(mask)
        mask_cellnbs = np.unique(mask_label)
        mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

        if len(mask_cellnbs) == 1:
            if contours[0].size < 6:
                outputs[outputs == outputs_cellnb] == 0
                outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != outputs_cellnb]
        else:
            for mask_cellnb in mask_cellnbs:
                mask_sc = mask_label == mask_cellnb

                contours, _ = cv2.findContours(mask_sc.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours[0].size < 6:
                    outputs[mask_label == mask_cellnb] == 0
                    mask_cellnbs = mask_cellnbs[mask_cellnbs != mask_cellnb]

            if len(mask_cellnbs) == 0:
                outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != outputs_cellnb]

    if reset:

        for inputs_cellnb in inputs_cellnbs:
            new_cell = np.array([inputs_cellnb,img_counter,img_counter,0],dtype=np.uint32)[None]
            if reset:
                meta_data = new_cell
                reset = False
            else:
                meta_data = np.concatenate((meta_data,new_cell),axis=0)
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
            # assert inputs_cellnb != 0, f'Inconsistent GT between previous and current frame for ROI {dataset_counter-1} Frame {framenb} \n{raw_data_fp.name}'
            if inputs_cellnb == 0:
                print(raw_data_fp.name)
            inputs[inputs_copy == inputs_cellnb] = track_gt_cellnb
            outputs[outputs_copy == inputs_cellnb] = track_gt_cellnb

    img_counter += 1

    max_cellnb = meta_data.shape[0]
    track_gt = np.copy(outputs).astype(np.uint16)

    outputs_cellnbs = np.unique(outputs)
    outputs_cellnbs = outputs_cellnbs[outputs_cellnbs != 0]

    for outputs_cellnb in outputs_cellnbs:
        mask = outputs == outputs_cellnb
        mask_label = label(mask)
        mask_cellnbs = np.unique(mask_label)
        mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

        mask = outputs == outputs_cellnb
        mask_label = label(mask)
        mask_cellnbs = np.unique(mask_label)
        mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

        # if len(mask_cellnbs) > 1:
        #     for mask_cellnb in mask_cellnbs:
        #         mask_sc = mask_label == mask_cellnb

        #         contours, _ = cv2.findContours(mask_sc.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         assert len(contours) == 1

        #         if contours[0].size < 6:
        #             mask_label[mask_label == mask_cellnb] == 0
        #             mask_cellnbs = mask_cellnbs[mask_cellnbs != mask_cellnb]

        if len(mask_cellnbs) > 2:
            mask_areas = np.array([(mask_label == mask_cellnbs[i]).sum() for i in range(len(mask_cellnbs))])
            indices = np.argsort(mask_areas)[-2:]
            mask_cellnbs = mask_cellnbs[indices]

        if len(mask_cellnbs) == 2:
            max_cellnb += 1
            track_gt[mask_label == mask_cellnbs[0]] = max_cellnb
            div_cell_1 = np.array([max_cellnb,img_counter,img_counter,outputs_cellnb],dtype=np.uint32)[None]
            max_cellnb += 1
            track_gt[mask_label == mask_cellnbs[1]] = max_cellnb
            div_cell_2 = np.array([max_cellnb,img_counter,img_counter,outputs_cellnb],dtype=np.uint32)[None]
            meta_data = np.concatenate((meta_data,div_cell_1,div_cell_2),axis=0)
        elif len(mask_cellnbs) > 2:
            raise NotImplementedError

        else:
            if outputs_cellnb in np.unique(inputs):
                meta_data[outputs_cellnb-1,2] = img_counter 
            else:
                max_cellnb += 1
                track_gt[mask_label == mask_cellnbs[0]] = max_cellnb
                new_cell = np.array([max_cellnb,img_counter,img_counter,0],dtype=np.uint32)[None]
                meta_data = np.concatenate((meta_data,new_cell),axis=0)

    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}' / f't{img_counter:03d}.tif'),img)
    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / f'man_track{img_counter:03d}.tif'),track_gt)
    cv2.imwrite(str(ctc_path / f'{dataset_counter:02d}_GT' / 'SEG' / f'man_seg{img_counter:03d}.tif'),track_gt)

    old_dataset = dataset

np.savetxt(ctc_path / f'{dataset_counter:02d}_GT' / 'TRA' / ('man_track.txt'),meta_data,fmt='%d')