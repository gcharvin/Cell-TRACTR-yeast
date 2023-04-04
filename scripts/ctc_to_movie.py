from pathlib import Path
import numpy as np
import re
import cv2
import ffmpeg

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test')
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/ctc')
# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results/230401_moma_track_two_stage_dn_enc_dn_track_dab_mask/test')
raw_datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test')

dataset_ids = [dataset_id for dataset_id in datapath.iterdir() if re.findall('\d\d$',dataset_id.name)]

all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(1000)])
colors = {}
alpha = 0.3

for dataset_id in dataset_ids:
    man_track = np.loadtxt(dataset_id / 'res_track.txt',dtype=np.uint16)
    filepaths = sorted([filepath for filepath in dataset_id.iterdir() if filepath.suffix == '.tif'])
    imgpaths = sorted([imgpath for imgpath in (raw_datapath / dataset_id.name).iterdir() if re.findall('t\d\d\d',imgpath.stem)])
    movie = []

    for idx,filepath in enumerate(filepaths):

        instance = cv2.imread(str(filepath),cv2.IMREAD_ANYDEPTH)
        img = cv2.imread(str(imgpaths[idx]),cv2.IMREAD_ANYDEPTH)
        img = np.stack((img,img,img),axis=-1)

        cellnbs = np.unique(instance)
        cellnbs = cellnbs[cellnbs != 0]
        daus = []
        for cellnb in cellnbs:
            if man_track[cellnb-1,1] == idx and man_track[cellnb-1,-1] > 0:
                mother_id = man_track[cellnb-1,-1]
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
            img = cv2.putText(
                img,
                text = str(cellnb), 
                org=(max(cell_loc[1] - (img.shape[1] // 3) * int(np.log10(cellnb)),0), cell_loc[0]), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale,
                color = (0,0,0),
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