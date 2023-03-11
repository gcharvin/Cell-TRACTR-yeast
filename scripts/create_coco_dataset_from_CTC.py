import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import re 
import utils_coco as utils

mothermachine = True

if mothermachine:
    min_area = 30
    dataset = 'moma'
    target_size = (256,32)
else:
    min_area = 10
    dataset = '2D'
    target_size = (256,256)

licenses = [{'id': 1,'name': 'MIT license', 'url':'add later'}]
categories = [{'id': 1, 'name': 'cell'}]
info = {
    'contributor': 'Owen OConnor',
    'date_created':'2022',
    'description':'E Coli cells growing in mother machine',
    'version': '1.0',
    'year': '2023'
    }

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data') / dataset
folders = ['train','val']
utils.create_folders(datapath,folders)

img_reader = utils.reader(mothermachine=mothermachine, target_size=target_size, min_area=min_area)

ctc_folders = [x for x in (datapath / 'CTC').iterdir() if x.is_dir() and 'GT' not in str(x)]
train_sets, val_sets = utils.train_val_split(ctc_folders,split=0.8)

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []

    if mothermachine:
        shifts = [0]
    else:
        shifts = [0,50,100]

    for dataset_path in dataset_paths:

        fps = sorted(dataset_path.glob('*.tif'))
        dataset_name = dataset_path.name

        with open(dataset_path.parent / (dataset_name + '_GT') / 'TRA' / 'man_track.txt') as f:
            track_file = []
            for line in f:
                line = line.split() # to deal with blank 
                if line:            # lines (ie skip them)
                    line = [int(i) for i in line]
                    track_file.append(line)
            track_file = np.stack(track_file)

        for counter,fp in enumerate(tqdm(fps)):

            fn = fp.name

            for shift in shifts:
                if shift == 0:
                    num_shifts = [[shift,shift]]
                else:
                    num_shifts = [[-shift,-shift],[shift,shift],[-shift,shift],[shift,-shift],[-shift,0],[0,-shift],[shift,0],[0,shift]]

                for shift_frame in num_shifts:
                    if not mothermachine:
                        prev_inputs = cv2.imread(str(datapath / 'raw_data' / 'inputs' / fn),cv2.IMREAD_ANYDEPTH)
                        img_reader.get_slices((prev_inputs > 0) * 1, shift_frame)

                    img = img_reader.read_image(fp)
                    gt = img_reader.read_gt(fp,counter,track_file)

                    cellnbs = np.unique(gt)
                    cellnbs = cellnbs[cellnbs != 0]

                    if len(cellnbs) > 0:
                        for cellnb in cellnbs:
                            annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name)
                            annotations.append(annotation)
                            annotation_id += 1 
                    else:
                        annotation = utils.create_anno(gt,-1,image_id,annotation_id,dataset_name)
                        annotations.append(annotation)
                        annotation_id += 1 


                    if not mothermachine:
                        fn = Path(fn).stem + f'_shift_{shift:03d}_{shift_frame[0]}_{shift_frame[1]}' + Path(fn).suffix
                    else:
                        fn = dataset_name +'_' + fn

                    image = {
                        'license': 1,
                        'file_name': fn,
                        'height': img.shape[0],
                        'width': img.shape[1],
                        'id': image_id,
                        'frame_id': 0,
                        'seq_length': 1,
                        }    

                    images.append(image)

                    image_id += 1

                    cv2.imwrite(str(datapath / folder / 'img' / fn),img)
                    cv2.imwrite(str(datapath / folder / 'gt' / fn),gt)

        np.savetxt(datapath / 'man_track' / (dataset_name + '.txt'),track_file,fmt='%d')
            
    metadata = {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
        'sequences': 'cells',
    }
        
    with open(datapath / 'annotations' / (f'{folder}') / (f'anno.json'), 'w') as f:
        json.dump(metadata,f, cls=utils.NpEncoder)
                
