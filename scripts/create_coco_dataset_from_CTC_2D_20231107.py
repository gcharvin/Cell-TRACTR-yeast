import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import utils_coco as utils
import re
from skimage.measure import label

dataset = '2D' # ['moma',''2D','DIC-C2DH-HeLa','Fluo-N2DH-SIM+']

if dataset == 'moma':
    min_area = 30
    target_size = (256,32)
elif dataset == '2D':
    min_area = 30
    target_size = (256,256)
elif dataset == '2D_512x512':
    min_area = 30
    target_size = (512,512)
elif dataset == 'DIC-C2DH-HeLa':
    min_area = 30
    target_size = (512,512)
elif dataset == 'Fluo-N2DH-SIM+':
    min_area = 30
    target_size = (512,512)
else:
    raise NotImplementedError

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

img_reader = utils.reader(dataset=dataset, target_size=target_size, min_area=min_area)

ctc_folders = sorted([x for x in (datapath / 'CTC').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])

if dataset in ['moma','2D','2D_512x512']:
    train_sets, val_sets = utils.train_val_split(ctc_folders,split=0.8)
else:
    train_sets = ctc_folders[:2]
    val_sets = ctc_folders[2:]

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []
    skip = 0

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

        img_reader.load_track_file(track_file)

        for counter,fp in enumerate(tqdm(fps)):
            
            fn = fp.name
            fn_orig = fn
            framenb = int(re.findall('\d+',fn)[-1])
            
            img = img_reader.read_image(fp)
            gt = img_reader.read_gt(fp,counter)

            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]

            framenb = int(re.findall('\d+',fp.name)[-1])
            man_track = track_file[(track_file[:,1] <= framenb) * (track_file[:,2] >= framenb)]

            np.array_equal(sorted(cellnbs),man_track[:,0])
        
            if len(cellnbs) == 0:
                cellnbs = [-1]

            for cellnb in cellnbs:
                    mask_sc = label(gt == cellnb)
                    annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name)
                    annotations.append(annotation)
                    annotation_id += 1 

            fn = f'CTC_{dataset_name}_frame_{framenb:03d}.tif'

            image = {
                'license': 1,
                'file_name': fn,
                'height': img.shape[0],
                'width': img.shape[1],
                'id': image_id,
                'ctc_id': fp.parent.name,
                'frame_id': 0,
                'seq_length': 1,
                }    

            images.append(image)

            image_id += 1

            cv2.imwrite(str(datapath / folder / 'img' / fn),img)
            cv2.imwrite(str(datapath / folder / 'gt' / fn),gt)

        np.savetxt(datapath / 'man_track' / (f'{dataset_name}.txt'),track_file,fmt='%d')

    metadata = {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
        'sequences': 'cells',
        'max_num_of_cells': img_reader.max_num_of_cells,
    }
        
    with open(datapath / 'annotations' / (f'{folder}') / (f'anno.json'), 'w') as f:
        json.dump(metadata,f, cls=utils.NpEncoder)

    print(f'Max number of cells in all frames is {img_reader.max_num_of_cells}')
    print(f'{skip:03d} folders skipped!')
                
