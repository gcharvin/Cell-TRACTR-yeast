from pathlib import Path  
import re
import math
import cv2
from tqdm import tqdm
import numpy as np
datapath = Path('/projectnb/dunlop/ooconnor/16bit/cell-trackformer/data/cells')

trainingsets = []#['train','val']
remove = False
target_size = (256,32)

subfolders = [['prev_cur','prev'],['cur','fut_prev']]

for trainingset in trainingsets:
    for subfolder in subfolders:

        fps = sorted(list((datapath / trainingset / (subfolder[0] + '_gt')).glob('*.png')))

        for fp in fps:
            remove_file = False
            prev_outputs = cv2.imread(str(fp),cv2.IMREAD_UNCHANGED)      
            # assert inputs.shape == target_size

            # inputs_cellnbs = np.unique(inputs)[1:]
            # for inputs_cellnb in inputs_cellnbs:
            #     if np.sum(inputs == inputs_cellnb) < 40:
            #         if np.max(np.where(inputs==inputs_cellnb)[0]) < 40:
            #             remove_file = True

            cur_inputs = cv2.imread(str(datapath / trainingset / (subfolder[1] + '_gt') / fp.name),cv2.IMREAD_UNCHANGED)
            # assert outputs.shape == target_size

            if not ((prev_outputs > 0) == (cur_inputs > 0)).all() or remove_file:
                print(fp.name)
                if remove:
                    for folder in ['prev_prev','prev_cur','prev','cur','fut_prev','fut']:
                        for f in ['_gt','_img']:
                            if (datapath / trainingset / (folder + f) / fp.name).exists():
                                (datapath / trainingset / (folder + f) / fp.name).unlink()

                    for folder in ['img','previmg','inputs','outputs']:
                        if (datapath / 'raw_data' / folder / fp.name).exists():
                            (datapath / 'raw_data' / folder / fp.name).unlink()

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/moma/test/raw_data')

remove = False
target_size = (256,32)


fps = sorted(list((datapath / 'inputs').glob('*.png')))

for fp in tqdm(fps):
    remove_file = False
    framenb = re.findall('\d+',fp.stem)[-1]
    framenb_index = fp.name.index('Frame' + framenb)
    prevfn = fp.name[:framenb_index] + f'Frame{int(framenb)-1:06d}.png'

    if (datapath / 'outputs' / prevfn).exists():
        prev_outputs = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)      

        cur_inputs = cv2.imread(str(datapath / 'outputs' / prevfn),cv2.IMREAD_ANYDEPTH)

        if not ((prev_outputs > 0) == (cur_inputs > 0)).all() or remove_file:
            print(fp.name)
            if remove:
                for folder in ['prev_prev','prev_cur','prev','cur','fut_prev','fut']:
                    for f in ['_gt','_img']:
                        if (datapath / trainingset / (folder + f) / fp.name).exists():
                            (datapath / trainingset / (folder + f) / fp.name).unlink()

                for folder in ['img','previmg','inputs','outputs']:
                    if (datapath / 'raw_data' / folder / fp.name).exists():
                        (datapath / 'raw_data' / folder / fp.name).unlink()
