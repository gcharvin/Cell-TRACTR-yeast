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

ctc_id_dict = {}
count = 0
for ctc_folder in ctc_folders:
    img_paths = sorted(ctc_folder.iterdir())
    for img_path in img_paths:
        ctc_id_dict[ctc_folder.stem + '_' + img_path.stem] = count
        count += 1

if dataset in ['moma','2D','2D_512x512']:
    train_sets, val_sets = utils.train_val_split(ctc_folders,split=0.8)
else:
    train_sets = ctc_folders[:2]
    val_sets = ctc_folders[2:]

save_as_CTC = True
CTC_coco_folder = 'CTC_coco'

if save_as_CTC:
    (datapath / CTC_coco_folder).mkdir(exist_ok=True)

ctc_counter = 0

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []
    skip = 0

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

        img_reader.load_track_file(track_file)

        split_fps_list = []

        split_size = 10

        for split in range(np.ceil(len(fps) / split_size).astype(int)):
            split_fps_list.append([fp for fp in fps[split * split_size : (split+1) * split_size]])

        for f,split_fps in enumerate(split_fps_list):

            ann_length = len(annotations)

            if len(split_fps) < 5:
                continue

            print(split_fps[0].name,split_fps[-1].name)

            img_reader.reset_track_file()
            final_outputs = cv2.imread(str(split_fps[-1].parents[1] / (split_fps[-1].parent.name + '_GT') / 'TRA' / ('man_track' + split_fps[-1].stem[-3:] + split_fps[-1].suffix)),cv2.IMREAD_ANYDEPTH)

            num_cells = len(np.unique(final_outputs))

            if num_cells < 3:
                if np.random.random() > 0.1:
                    skip += 1
                    print(f'Skip! Only {len(np.unique(final_outputs)):02d} cells in the last frame')
                    continue
            elif num_cells < 10:
                if np.random.random() > 0.25:
                    skip += 1
                    print(f'Skip! Only {len(np.unique(final_outputs)):02d} cells in the last frame')
                    continue
            elif num_cells < 25:
                if np.random.random() > 0.5:
                    skip += 1
                    print(f'Skip! Only {len(np.unique(final_outputs)):02d} cells in the last frame')
                    continue
            else:
                print(f'There are {num_cells:03d} cells in the last frame of this movie')

            ctc_counter += 1

            print(f'working on CTC: {dataset_name} video {ctc_counter:03d}')

            if save_as_CTC:
                (datapath / CTC_coco_folder / f'{ctc_counter:03d}').mkdir(exist_ok=True)
                (datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT').mkdir(exist_ok=True)
                (datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'TRA').mkdir(exist_ok=True)
                (datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'SEG').mkdir(exist_ok=True)

            for counter,fp in enumerate(tqdm(split_fps)):
                
                fn = fp.name
                fn_orig = fn
                framenb = int(re.findall('\d+',fn)[-1])

                for shift in shifts:
                    if shift == 0:
                        num_shifts = [[shift,shift]]
                    else:
                        num_shifts = [[-shift,-shift],[shift,shift],[-shift,shift],[shift,-shift],[-shift,0],[0,-shift],[shift,0],[0,shift]]

                    for shift_frame in num_shifts:
                        if img_reader.crop:
                            img_reader.get_slices((final_outputs > 0) * 1, shift_frame)
                        
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
                                annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name,ctc_counter)
                                annotations.append(annotation)
                                annotation_id += 1 

                        fn = f'CTC_{dataset_name}_split_{f:02d}_frame_{framenb:03d}.tif'#_{counter:03d}.tif'

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

                        if save_as_CTC:
                            cv2.imwrite(str(datapath / CTC_coco_folder / f'{ctc_counter:03d}' / fn_orig),img)
                            cv2.imwrite(str(datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'TRA' / fn_orig),gt)
                            cv2.imwrite(str(datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'SEG' / fn_orig),gt)

            num_imgs = len(list((datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'TRA').glob('*.tif'))) 

            if num_imgs > 10:
                print(num_imgs)
                pass
            annotations = img_reader.clean_track_file(datapath / folder, dataset_name,split_fps,CTC_coco_folder, f, ctc_counter,ann_length,annotations)

            np.savetxt(datapath / 'man_track' / (f'{dataset_name}_{f:02d}.txt'),img_reader.crop_track_file,fmt='%d')

            if save_as_CTC:
                np.savetxt(datapath / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'TRA' / 'man_track.txt',img_reader.crop_track_file,fmt='%d')

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
                
