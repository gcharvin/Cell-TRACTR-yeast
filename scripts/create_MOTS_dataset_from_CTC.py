import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import re
from skimage.measure import label
from pycocotools import mask as m

dataset = 'moma' # ['moma',''2D','DIC-C2DH-HeLa','Fluo-N2DH-SIM+']
data_format = 'MOTS' # ['MOT','MOTS']

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data') / dataset / 'test' 
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results/230831_moma_flex_div_CoMOT_track_two_stage_dn_enc_dn_track_dn_track_group_dab_intermediate_mask_4_enc_4_dec_layers/test')
# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test')
# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/test')

hotapath = datapath / 'HOTA'
hotapath.mkdir(exist_ok=True)

ctc_folders = sorted([x for x in (datapath / 'CTC').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])
    
for ctc_folder in ctc_folders:
    
    if (ctc_folder.parent / (ctc_folder.name + '_GT')).exists():
        fps = sorted((ctc_folder.parent / (ctc_folder.name + '_GT') / 'TRA').glob('*.tif'))
        man_track_path = ctc_folder.parent / (ctc_folder.name + '_GT') / 'TRA' / 'man_track.txt'

        (hotapath / ctc_folder.name).mkdir(exist_ok=True)
        (hotapath / ctc_folder.name / 'gt').mkdir(exist_ok=True)

        if (hotapath / ctc_folder.name / 'gt' / 'gt.txt').exists():
            (hotapath / ctc_folder.name / 'gt' / 'gt.txt').unlink()

    else:
        tracker = datapath.parts[5]
        fps = sorted(ctc_folder.glob('*.tif'))
        man_track_path = ctc_folder / 'res_track.txt'

        (hotapath / tracker).mkdir(exist_ok=True)
        if (hotapath / tracker / f'{ctc_folder.name}.txt').exists():
            (hotapath / tracker / f'{ctc_folder.name}.txt').unlink()


    with open(man_track_path) as f:
        track_file = []
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [int(i) for i in line]
                track_file.append(line)
        track_file = np.stack(track_file)

    for counter,fp in enumerate(tqdm(fps)):

        gt = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)

        cellnbs = np.unique(gt)
        cellnbs = cellnbs[cellnbs != 0]

        framenb = int(re.findall('\d+',fp.name)[-1])
        man_track = track_file[(track_file[:,1] <= framenb) * (track_file[:,2] >= framenb)]
    
        # If no cell is present in image, we skip it
        if len(cellnbs) == 0:
            continue

        for cellnb in cellnbs:

            mask = gt == cellnb
            parent = track_file[cellnb-1,-1] if track_file[cellnb-1,1] == framenb and track_file[cellnb-1,-1] > 0 else 0

            if data_format == 'MOTS':
                rle = m.encode(np.asfortranarray(mask))['counts'].decode("utf-8")
                line = f'{counter+1} {cellnb} {1} {gt.shape[0]} {gt.shape[1]} {rle} {parent}'

            elif data_format == 'MOT':
                y, x = np.where(mask != 0)
                width = (np.max(x) - np.min(x) ) 
                height = (np.max(y) - np.min(y)) 
                bbox = (np.min(x) ,np.min(y),width,height)
                line = f'{counter+1} {cellnb} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} -1 -1 -1 -1 {parent}'
            else:
                raise NotImplementedError

            if (ctc_folder.parent / (ctc_folder.name + '_GT')).exists():
                with open(hotapath / ctc_folder.name / 'gt' / 'gt.txt', 'a') as file:
                    file.write(line + '\n')
            else:
                with open(hotapath / tracker / f'{ctc_folder.name}.txt', 'a') as file:
                    file.write(line + '\n')


        

