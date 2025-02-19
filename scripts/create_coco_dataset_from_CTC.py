import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import utils_coco as utils
import re
from skimage.measure import label

dataset_name = 'DynamicNuclearNet-tracking-v1_0'
basepath = Path('/projectnb/dunlop/ooconnor/MOT/data') 
cocopath = basepath / dataset_name / 'COCO'
CTC_datasets = ['DynamicNuclearNet-tracking-v1_0'] #['moma','DynamicNuclearNet-tracking-v1_0', '2D','MLCI']
# CTC_datasets = []
track = True

min_area = 0
target_size = None
resize = False

if min_area > 0 and track:
    raise NotImplementedError # 'Need to fix divisions if cell removed had just divided'

licenses = [{'id': 1,'name': 'MIT license'}]
categories = [{'id': 1, 'name': 'cell'}]
info = ''

cocopath.mkdir(exist_ok=True,parents=True)
folders = ['train','val']
utils.create_folders(cocopath,folders,track, clear_old_data=False)

img_reader = utils.reader(track=track, target_size=target_size, resize=resize, min_area=min_area)

if len(CTC_datasets) > 0:
    datasets = [basepath / 'CTC_datasets' / CTC_dataset for CTC_dataset in CTC_datasets if (basepath / 'CTC_datasets' / CTC_dataset / 'CTC' / 'train').exists()]
else:
    datasets = [basepath / 'CTC_datasets' / CTC_dataset for CTC_dataset in (basepath / 'CTC_datasets').iterdir() if (basepath / 'CTC_datasets' / CTC_dataset / 'CTC' / 'train').exists()]

train_sets = []
val_sets = []

for dataset in datasets:
    train_sets += sorted([x for x in (dataset / 'CTC' / 'train').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])
    val_sets += sorted([x for x in (dataset / 'CTC' / 'val').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])

dataset_count_dict = {'train':{'total': 0},'val':{'total':0}}

for name, sets in zip(['train','val'],[train_sets,val_sets]):

    for set in sets:
        fps = list(set.iterdir())

        dataset_name = set.parents[2].name

        if dataset_name not in dataset_count_dict[name]:
            dataset_count_dict[name][dataset_name] = len(fps)
        else:
            dataset_count_dict[name][dataset_name] += len(fps)

        dataset_count_dict[name]['total'] += len(fps)

# Save the dictionary
with open(str(cocopath / 'counts.json'), 'w') as f:
    json.dump(dataset_count_dict, f)

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    print(f'Loading {folder} folders...')

    # if folder == 'val':
    #     continue
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []
    skip = 0
    man_track_id = 1
    counter_a = 0

    for d,dataset_path in enumerate(dataset_paths):

        fps = sorted(dataset_path.glob('*.tif'))

        if track and len(fps) < 4: # You need a minimum of 4 frames for flexible divisions while training. 3 frames if not using flexible divisons (prev_prev, prev, cur and fut target)
            continue

        # if int(dataset_path.name) < 89:
        #     continue

        dataset_num = dataset_path.name
        dataset_name = dataset_path.parents[2].name
        print(f'Working on dataset {dataset_name} ({d+1}/{len(dataset_paths)})')

        if track:
            with open(dataset_path.parent / (dataset_num + '_GT') / 'TRA' / 'man_track.txt') as f:
                track_file = []
                for line in f:
                    line = line.split() # to deal with blank 
                    if line:            # lines (ie skip them)
                        line = [int(i) for i in line]
                        track_file.append(line)
                track_file = np.stack(track_file)

            img_reader.load_track_file(track_file)

        img_reader.read_all_gts(fps)

        for counter,fp in enumerate(tqdm(fps)):
            
            fn = fp.name
            framenb = int(re.findall('\d+',fn)[-1])
            
            img = img_reader.read_image(fp)
            gt = img_reader.read_gt(fp,counter)

            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]

            if track:
                track_file = img_reader.track_file_orig
                man_track = track_file[(track_file[:,1] <= framenb) * (track_file[:,2] >= framenb)]
                np.array_equal(sorted(cellnbs),man_track[:,0])
        
            if len(cellnbs) == 0:
                cellnbs = [-1]

            for cellnb in cellnbs:
                mask_sc = label(gt == cellnb)
                annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name,dataset_num)
                annotations.append(annotation)
                annotation_id += 1 
                assert annotation['track_id'] in track_file[:,0]
                row = track_file[track_file[:,0] == annotation['track_id']][0]
                assert row[1] <= counter
                assert row[2] >= counter

            fn = f'{dataset_name}_CTC_{dataset_num}_frame_{framenb:03d}.tif'
            
            image = {
                'license': 1,
                'file_name': fn,
                'height': img.shape[0],
                'width': img.shape[1],
                'id': image_id,
                'ctc_id': dataset_num,
                'dataset_name': dataset_name,
                'man_track_id': man_track_id,
                'frame_id': 0,
                'seq_length': 1,
                }    

            images.append(image)

            image_id += 1

            # cv2.imwrite(str(cocopath / folder / 'img' / fn),img)
            # cv2.imwrite(str(cocopath / folder / 'gt' / fn),gt)

        if track:
            np.savetxt(cocopath / 'man_track' / folder / (f'{man_track_id}.txt'),track_file,fmt='%d')
            man_track_id += 1

        for idx,a in enumerate(tqdm(annotations)):
            if idx < counter_a:
                continue
            img_id = a['image_id']
            image = images[img_id]
            framenb = int(re.findall('\d+', image['file_name'])[-1])
            dataset_num = a['dataset_num']
            # man_track_id = image['man_track_id']
            assert a['dataset_num'] == image['ctc_id']
            # with open(str(cocopath / 'man_track' / folder / (f'{man_track_id}.txt'))) as f:
            #     track_file = []
            #     for line in f:
            #         line = line.split() # to deal with blank 
            #         if line:            # lines (ie skip them)
            #             line = [int(i) for i in line]
            #             track_file.append(line)
            #     track_file = np.stack(track_file)

            assert a['track_id'] in track_file[:,0]
            row = track_file[track_file[:,0] == a['track_id']][0]
            assert row[1] <= framenb
            assert row[2] >= framenb

        counter_a = len(annotations)

    metadata = {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
        'sequences': 'cells',
        'max_num_of_cells': img_reader.max_num_of_cells,
    }
        
    # with open(cocopath / 'annotations' / folder / (f'anno.json'), 'w') as f:
    #     json.dump(metadata,f, cls=utils.NpEncoder)

                
