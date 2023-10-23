from pathlib import Path
import numpy as np
import re
import cv2
import ffmpeg
import tifffile

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test/CTC')
# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/test/CTC')
# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results/230504_moma_track_two_stage_dn_enc_dn_track_dn_object_dab_mask/test')
# raw_datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test')

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/CTC/')
# raw_datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/DIC-C2DH-HeLa/test')

dataset_ids = sorted([dataset_id for dataset_id in datapath.iterdir() if re.findall('\d\d$',dataset_id.name)])

all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(1000)])
colors = {}
alpha = 0.3

for dataset_id in dataset_ids:
    # if dataset_id.name != '24':
    #     continue
    try:
        man_track = np.loadtxt(dataset_id / 'res_track.txt',dtype=np.uint16)
        filepaths = sorted([filepath for filepath in dataset_id.iterdir() if filepath.suffix == '.tif'])
        TRA_path = dataset_id / 'TRA_log.txt'
        imgpath = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test/CTC')
        img_fps = sorted([img_fp for img_fp in (imgpath / dataset_id.name).iterdir() if re.findall('\d\d\d$',img_fp.stem) and img_fp.suffix == '.tif'])
        print_errors = True
    except:
        man_track = np.loadtxt(dataset_id.parent / (dataset_id.name + '_GT') / 'TRA' / 'man_track.txt',dtype=np.uint16)
        filepaths = sorted([filepath for filepath in (dataset_id.parent / (dataset_id.name + '_GT') / 'TRA').iterdir() if filepath.suffix == '.tif'])
        maskpaths = sorted([filepath for filepath in (dataset_id.parent / (dataset_id.name + '_GT') / 'SEG').iterdir() if filepath.suffix == '.tif'])
        TRA_path = None
        img_fps = sorted([imgpath for imgpath in (datapath / dataset_id.name).iterdir() if re.findall('\d\d\d$',imgpath.stem) and imgpath.suffix == '.tif'])
        print_errors = False

    # We remove cells that disappear and reappear
    divisions = np.unique(man_track[:,-1])
    divisions = divisions[divisions != 0]
    for div in divisions:
        if (man_track[:,-1] == div).sum() == 1:
            man_track[man_track[:,-1] == div,-1] = 0
    
    movie = []

    max_pixel = 0
    for idx,img_fp in enumerate(img_fps):
        max_pixel = max(max_pixel, np.max(cv2.imread(str(img_fp),cv2.IMREAD_ANYDEPTH)))

    for idx,img_fp in enumerate(img_fps):

        instance = cv2.imread(str(filepaths[idx]),cv2.IMREAD_ANYDEPTH)
        # mask = tifffile.imread(maskpaths[idx])
        img = cv2.imread(str(img_fp),cv2.IMREAD_ANYDEPTH)
        img = np.stack((img,img,img),axis=-1)

        img = (img - np.min(img)) / max_pixel
        img = (img * 255).astype(np.uint8)

        cellnbs = np.unique(instance)
        cellnbs = cellnbs[cellnbs != 0]
        daus = []

        # for cellnb in cellnbs:
        #     replace = True
        #     mask_cellnbs = np.unique(mask[instance == cellnb])
        #     mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]
        #     if len(mask_cellnbs) > 1:
        #         if cellnb in mask_cellnbs:
        #             mask_cellnb = cellnb
        #         else:
        #             replace = False
        #     elif len(mask_cellnbs) == 1:
        #         mask_cellnb = mask_cellnbs[0]
        #     else:
        #         replace = False

        #     if replace:
        #         instance[mask == mask_cellnb] = cellnb

        for cellnb in cellnbs:
            cellnb_ind = man_track[:,0] == cellnb
            assert cellnb in man_track[:,0]
            if man_track[cellnb_ind,1] == idx and man_track[cellnb_ind,-1] > 0:
                mother_id = man_track[cellnb_ind,-1][0]
                dau_1, dau_2 = man_track[man_track[:,-1] == mother_id,0]
                daus.append([dau_1,dau_2])

                if cellnb == dau_1:
                    if np.where(instance == dau_1)[0].mean() < np.where(instance == dau_2)[0].mean():
                        colors[cellnb] = colors[mother_id]
                    else:
                        colors[cellnb] = all_colors[cellnb]
                elif cellnb == dau_2:
                    if np.where(instance == dau_2)[0].mean() < np.where(instance == dau_1)[0].mean():
                        colors[cellnb] = colors[mother_id]
                    else:
                        colors[cellnb] = all_colors[cellnb]
                else:
                    NotImplementedError
            else: 
                if cellnb not in colors.keys():
                    colors[cellnb] = all_colors[cellnb]
            
            color_mask = np.zeros_like(img)
            color_mask[instance == cellnb] = colors[cellnb]
            img[instance == cellnb] = alpha * color_mask[instance==cellnb] + (1-alpha) * img[instance == cellnb]

            cell_loc = np.where(instance == cellnb)
            cell_loc = [int(np.median(cell_loc[0])), int(np.median(cell_loc[1]))]

            fontscale = 0.4
            if img.shape[1] < 100:
                org = (max(cell_loc[1] - (img.shape[1] // 3) * int(np.log10(cellnb)),0), cell_loc[0])
            else:
                org = (cell_loc[1], cell_loc[0])
            img = cv2.putText(
                img,
                text = str(cellnb), 
                org=org, 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale,
                color = (0,0,0) if max_pixel < 256 else (255,255,255),
                thickness=1,
                )

        for div in daus:
            cell_1 = np.where(instance == div[0])
            cell_1 = [int(np.median(cell_1[0])), int(np.median(cell_1[1]))]
            cell_2 = np.where(instance == div[1])
            cell_2 = [int(np.median(cell_2[0])), int(np.median(cell_2[1]))]

            if cell_1[0] > cell_2[0]:
                cell_1, cell_2 = cell_2, cell_1

            img = cv2.arrowedLine(
                img,
                (cell_1[1], cell_1[0]),
                (cell_2[1], cell_2[0]),
                color=(0, 0, 0),
                thickness=1,
            )

        img = cv2.putText(
            img,
            text = str(idx), 
            org=(0,10), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = fontscale,
            color = (255,255,255),
            thickness=1,
            )

        movie.append(img)

    movie = np.stack(movie)

    if print_errors:
        movie_error = np.zeros_like(movie)
        for idx,img_fp in enumerate(img_fps):
            img = cv2.imread(str(img_fp),cv2.IMREAD_ANYDEPTH)
            img = np.stack((img,img,img),axis=-1)

            img = (img - np.min(img)) / max_pixel
            img = (img * 255).astype(np.uint8)

            movie_error[idx] = img

        error_dict = {}
        colors = {'SO': (255,0,0),'FP': (0,255,0), 'FN': (0,0,255)}
        if TRA_path.exists():
            with open(TRA_path, "r") as file:
                for line in file:
                    if 'Splitting Operations' in line:
                        error = 'SO'
                        error_dict['SO'] = []
                        continue
                    elif 'False Negative Vertices' in line:
                        error = 'FN'
                        error_dict['FN'] = []
                        continue
                    elif 'False Positive Vertices' in line:
                        error = 'FP'
                        error_dict['FP'] = []
                        continue
                    elif 'Redundant Edges To Be Deleted' in line:
                        break
                    
                    framenb, label = list(map(int,(re.findall('\d+',line))))

                    error_dict[error].append([framenb,label])

                for error in error_dict.keys():
                    framenbs_labels = error_dict[error]

                    for framenb,label in framenbs_labels:

                        if error in ['SO','FP']:
                            instance = cv2.imread(str(filepaths[framenb]),cv2.IMREAD_ANYDEPTH)
                            assert str(framenb) in filepaths[framenb].stem

                            # Find contours in the mask
                            contours, _ = cv2.findContours((instance == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            x,y,w,h = cv2.boundingRect(contours[0])

                            cv2.rectangle(movie_error[framenb], (x, y), (x + w, y + h), colors[error], 2)

                        else:
                            instance = cv2.imread(str(img_fps[0].parents[1] / (dataset_id.name + '_GT') / 'TRA' / f'man_track{framenb:03d}.tif'),cv2.IMREAD_ANYDEPTH)
                            assert str(framenb) in img_fps[framenb].stem

                            mask = instance == label
                            mask_color = np.zeros((mask.shape + (3,)))
                            mask_color[mask] = colors[error]
                            movie_error[framenb][mask] = movie_error[framenb][mask] * 0.5 + mask_color[mask] * 0.5

        movie = np.concatenate((movie,movie_error),axis=2)



    height, width, _ = movie[0].shape
    crf=20
    verbose = 1
    filename = dataset_id / 'ctc_movie.mp4'
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    quiet = [] if verbose else ["-loglevel", "error", "-hide_banner"]
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=7,
        )
        .output(
            str(filename),
            pix_fmt="yuv420p",
            vcodec="libx264",
            crf=crf,
            preset="veryslow",
        )
        .global_args(*quiet)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write frames:
    for frame in movie:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()