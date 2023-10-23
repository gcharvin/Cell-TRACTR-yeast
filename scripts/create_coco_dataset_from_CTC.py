import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import utils_coco as utils
import re

dataset = 'moma' # ['moma',''2D','DIC-C2DH-HeLa','Fluo-N2DH-SIM+']

if dataset == 'moma':
    min_area = 30
    target_size = (256,32)
elif dataset == '2D':
    min_area = 10
    target_size = (256,256)
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

ctc_id_dict = {}
count = 0
for ctc_folder in ctc_folders:
    img_paths = sorted(ctc_folder.iterdir())
    for img_path in img_paths:
        ctc_id_dict[ctc_folder.stem + '_' + img_path.stem] = count
        count += 1

if dataset in ['moma','2D']:
    train_sets, val_sets = utils.train_val_split(ctc_folders,split=0.8)
else:
    train_sets = ctc_folders[:2]
    val_sets = ctc_folders[2:]

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []

    if dataset != '2D':
        shifts = [0]
    else:
        rand_num = np.random.randint(3)
        shifts = [0,50,100][rand_num:rand_num+1]
        shifts = [0]

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
                    if img_reader.crop:
                        # prev_inputs = cv2.imread(str(datapath / 'raw_data' / 'inputs' / fn),cv2.IMREAD_ANYDEPTH)
                        final_outputs = cv2.imread(str(fp.parents[1] / (fp.parent.name + '_GT') / 'TRA' / ('man_track' + fps[-1].stem[-3:] + fp.suffix)),cv2.IMREAD_ANYDEPTH)
                        img_reader.get_slices((final_outputs > 0) * 1, shift_frame)
                    
                    img = img_reader.read_image(fp)
                    gt = img_reader.read_gt(fp)

                    cellnbs = np.unique(gt)
                    cellnbs = cellnbs[cellnbs != 0]

                    framenb = int(re.findall('\d+',fp.name)[-1])
                    man_track = track_file[(track_file[:,1] <= framenb) * (track_file[:,2] >= framenb)]

                    np.array_equal(sorted(cellnbs),man_track[:,0])
                
                    if len(cellnbs) == 0:
                        cellnbs = [-1]

                    for cellnb in cellnbs:
                            annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name)
                            annotations.append(annotation)
                            annotation_id += 1 

                    fn = dataset_name +'_' + fn

                    image = {
                        'license': 1,
                        'file_name': fn,
                        'height': img.shape[0],
                        'width': img.shape[1],
                        'id': image_id,
                        # 'ctc_id': ctc_id_dict[Path(fn).stem],
                        'ctc_id': fp.parent.name,
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
                
