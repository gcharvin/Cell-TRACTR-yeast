from pathlib import Path
import numpy as np
import re
import cv2
import ffmpeg
import pickle

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test')
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/test')
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results/230831_moma_no__flex_div_CoMOT_track_two_stage_dn_enc_dn_track_dn_track_group_dab_intermediate_mask_4_enc_4_dec_layers/test')
ctc_path = datapath / 'CTC'
hota_path = datapath / 'HOTA' / datapath.parts[5]
gt_path = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test/CTC')
save_path = datapath / 'movies'

flex_div = True
flex_div = '_flex_div' if flex_div else ''

dataset_paths = sorted([dataset_id for dataset_id in ctc_path.iterdir() if re.findall('\d\d$',dataset_id.name)])

all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(1000)])
colors = {}
transparency = 0.4
fontscale = 0.4

display_ctc_errors = True
display_hota_errors = True
display_FP_FN_pixels = True

if hota_path.exists():
    with open(hota_path / ('cell_data' + flex_div + '.pkl'), 'rb') as f:
        hota_errors = pickle.load(f)

if not (save_path).exists():
    (save_path).mkdir()

for dataset_path in dataset_paths:

    dataset_id = dataset_path.name

    man_track_pred = np.loadtxt(dataset_path / 'res_track.txt',dtype=np.uint16)
    filepaths_pred = sorted([filepath for filepath in dataset_path.iterdir() if filepath.suffix == '.tif'])
    TRA_path = dataset_path / 'TRA_log.txt'
    img_fps = sorted([img_fp for img_fp in (gt_path / dataset_id).iterdir() if re.findall('\d\d\d$',img_fp.stem)])

    man_track_gt = np.loadtxt(gt_path / (dataset_id + '_GT') / 'TRA' / 'man_track.txt',dtype=np.uint16)
    filepaths_gt = sorted([filepath for filepath in (gt_path / (dataset_id + '_GT') / 'TRA').iterdir() if filepath.suffix == '.tif'])
    maskpaths_gt = sorted([filepath for filepath in (gt_path / (dataset_id + '_GT') / 'SEG').iterdir() if filepath.suffix == '.tif'])
    imgpaths_gt = sorted([gt_path for gt_path in (gt_path / dataset_id).iterdir() if re.findall('\d\d\d$',gt_path.stem)])

    if not (save_path / dataset_id).exists():
        (save_path / dataset_id).mkdir()

    # We remove cells that disappear and reappear
    divisions = np.unique(man_track_pred[:,-1])
    divisions = divisions[divisions != 0]
    for div in divisions:
        if (man_track_pred[:,-1] == div).sum() == 1:
            man_track_pred[man_track_pred[:,-1] == div,-1] = 0
    
    movies = []
    movie_bf = []
    movie_bf_display = []

    max_pixel = 0
    for idx,img_fp in enumerate(img_fps):
        img = cv2.imread(str(img_fp),cv2.IMREAD_ANYDEPTH)
        img = np.repeat(img[:,:,None],3,axis=-1)
        movie_bf.append(img)        
        
        img_copy = np.concatenate((np.zeros((60,img.shape[1],3)),img),axis=0)

        img_copy = cv2.putText(
            img_copy,
            text = 'Raw', 
            org=(0,54), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = fontscale,
            color = (255,255,255),
            thickness=1,
            )
        
        movie_bf_display.append(img_copy)
        
    movie_bf = np.stack(movie_bf)
    movie_bf_display = np.stack(movie_bf_display)

    movies.append(movie_bf_display)
    
    for filepaths,man_track,dataset_name in [[filepaths_gt,man_track_gt,'GT'],[filepaths_pred,man_track_pred,'Pred']]:

        movie = [] 

        for idx,filepath in enumerate(filepaths):

            instance = cv2.imread(str(filepath),cv2.IMREAD_ANYDEPTH)

            img = movie_bf[idx].copy()

            cellnbs = np.unique(instance)
            cellnbs = cellnbs[cellnbs != 0]
            daus = []

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
                img[instance == cellnb] = transparency * color_mask[instance==cellnb] + (1-transparency) * img[instance == cellnb]

                cell_loc = np.where(instance == cellnb)
                cell_loc = [int(np.median(cell_loc[0])), int(np.median(cell_loc[1]))]

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

            movie.append(img)

        movie = np.stack(movie)   
        movie = np.concatenate((np.zeros((movie.shape[0],60,movie.shape[2],3)),movie),axis=1)


        for mov in movie:
            mov = cv2.putText(
                mov,
                text = dataset_name, 
                org=(0,54), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale+0.1,
                color = (255,255,255),
                thickness=1,
                )

        movies.append(np.stack(movie))

        if dataset_name == 'Pred' and display_ctc_errors:
            movie_ctc_Det_FP = movie_bf.copy()
            movie_ctc_Det_FN = movie_bf.copy()
            movie_ctc_Det_NS = movie_bf.copy()
            movie_ctc_Tra_EA = movie_bf.copy()
            movie_ctc_Tra_ED = movie_bf.copy()
            movie_ctc_Tra_EC = movie_bf.copy()

            color = (255,0,0)
            ctc_transparency = 0

            if TRA_path.exists():
                with open(TRA_path, "r") as file:
                    for line in file:
                        if 'Splitting Operations' in line:
                            error = 'NS'
                            continue
                        elif 'False Negative Vertices' in line:
                            error = 'FN'
                            continue
                        elif 'False Positive Vertices' in line:
                            error = 'FP'
                            continue
                        elif 'Redundant Edges To Be Deleted' in line:
                            error = 'ED'
                            continue
                        elif 'Edges To Be Added' in line:
                            error = 'EA'
                            continue
                        elif 'Edges with Wrong Semantics' in line:
                            error = 'EC'
                            continue
                        elif '=======' in line:
                            break
                        
                        if error in ['FP','FN','NS']:
                            framenb, label = list(map(int,(re.findall('\d+',line))))
                        elif error in ['EA','ED','EC']:
                            framenb_1, label_1, framenb_2, label_2 = list(map(int,(re.findall('\d+',line))))
                        else:
                            raise NotImplementedError

                        if error in ['NS','FP','ED','EC']:
                            if error in ['FP','NS']:
                                instance = cv2.imread(str(filepaths[framenb]),cv2.IMREAD_ANYDEPTH)
                            else:
                                instance_1 = cv2.imread(str(filepaths[framenb_1]),cv2.IMREAD_ANYDEPTH)
                                instance_2 = cv2.imread(str(filepaths[framenb_2]),cv2.IMREAD_ANYDEPTH)
                        elif error in ['FN','EA']:
                            if error in ['FN']:
                                instance = cv2.imread(str(gt_path / (dataset_id + '_GT') / 'TRA' / f'man_track{framenb:03d}.tif'),cv2.IMREAD_ANYDEPTH)
                            else:
                                instance_1 = cv2.imread(str(gt_path / (dataset_id + '_GT') / 'TRA' / f'man_track{framenb_1:03d}.tif'),cv2.IMREAD_ANYDEPTH)
                                instance_2 = cv2.imread(str(gt_path / (dataset_id + '_GT') / 'TRA' / f'man_track{framenb_2:03d}.tif'),cv2.IMREAD_ANYDEPTH)
                        else:
                            raise NotImplementedError

                        if error in ['FP','FN','NS']:
                            mask = instance == label
                            mask_color = np.zeros((mask.shape + (3,)))
                            mask_color[mask] = color
                            if error == 'FP':
                                movie_ctc_Det_FP[framenb][mask] = movie_ctc_Det_FP[framenb][mask] * ctc_transparency + mask_color[mask] * (1-ctc_transparency)
                            elif error == 'FN':
                                movie_ctc_Det_FN[framenb][mask] = movie_ctc_Det_FN[framenb][mask] * ctc_transparency + mask_color[mask] * (1-ctc_transparency)
                            elif error == 'NS':
                                movie_ctc_Det_NS[framenb][mask] = movie_ctc_Det_NS[framenb][mask] * ctc_transparency + mask_color[mask] * (1-ctc_transparency)
                        else:
                            mask_1 = instance_1 == label_1
                            mask_color_1 = np.zeros((mask_1.shape + (3,)))
                            mask_color_1[mask_1] = color
                            mask_2 = instance_2 == label_2
                            mask_color_2 = np.zeros((mask_2.shape + (3,)))
                            mask_color_2[mask_2] = color

                            if error == 'ED':
                                movie_ctc_Tra_ED[framenb_1][mask_1] = movie_ctc_Tra_ED[framenb_1][mask_1] * ctc_transparency + mask_color_1[mask_1] * (1-ctc_transparency)
                                movie_ctc_Tra_ED[framenb_2][mask_2] = movie_ctc_Tra_ED[framenb_2][mask_2] * ctc_transparency + mask_color_2[mask_2] * (1-ctc_transparency)
                            elif error == 'EA':
                                movie_ctc_Tra_EA[framenb_1][mask_1] = movie_ctc_Tra_EA[framenb_1][mask_1] * ctc_transparency + mask_color_1[mask_1] * (1-ctc_transparency)
                                movie_ctc_Tra_EA[framenb_2][mask_2] = movie_ctc_Tra_EA[framenb_2][mask_2] * ctc_transparency + mask_color_2[mask_2] * (1-ctc_transparency)
                            elif error == 'EC':
                                movie_ctc_Tra_EC[framenb_1][mask_1] = movie_ctc_Tra_EC[framenb_1][mask_1] * ctc_transparency + mask_color_1[mask_1] * (1-ctc_transparency)
                                movie_ctc_Tra_EC[framenb_2][mask_2] = movie_ctc_Tra_EC[framenb_2][mask_2] * ctc_transparency + mask_color_2[mask_2] * (1-ctc_transparency)

            ctc_movies = []

            for ctc_movie, text in zip([movie_ctc_Det_FP,movie_ctc_Det_FN,movie_ctc_Det_NS,movie_ctc_Tra_ED,movie_ctc_Tra_EA,movie_ctc_Tra_EC],['FP','FN','NS','ED','EA','EC']):

                ctc_movie = np.concatenate((np.zeros((ctc_movie.shape[0],60,ctc_movie.shape[2],3)),ctc_movie),axis=1)

                for img_ctc in ctc_movie:
                    img_ctc = cv2.putText(
                        img_ctc,
                        text = 'CTC', 
                        org=(0,30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = (255,255,255),
                        thickness=1,
                        )
                    
                    img_ctc = cv2.putText(
                        img_ctc,
                        text = text, 
                        org=(0,42), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = (255,255,255),
                        thickness=1,
                        )

                ctc_movies.append(ctc_movie)

            spacer = np.zeros((ctc_movie.shape[0],ctc_movie.shape[1],5,3))

            movies.append(spacer)
            movies.extend(ctc_movies)

        if dataset_name == 'Pred' and hota_path.exists() and display_hota_errors:
            
            movie_hota_Det_FN = movie_bf.copy()
            movie_hota_Det_FP = movie_bf.copy()
            
            movie_hota_Div_FN = movie_bf.copy()
            movie_hota_Div_FP = movie_bf.copy()

            movie_hota_Ass_Re = movie_bf.copy()
            movie_hota_Ass_Pr = movie_bf.copy()

            dataset_errors = hota_errors[dataset_id]

            errors = dataset_errors.keys()

            alpha = 9
            alpha_values = np.arange(0.05, 0.99, 0.05)

            for error in errors:

                dataset_error = dataset_errors[error]

                if 'FP' or 'Pr' in error:
                    color = (255,0,0)
                else:
                    color = (0,255,0)

                if 'Ass' in error:

                    dataset_error_alpha = dataset_error[alpha]

                    if len(dataset_error_alpha[0][0])>0:

                        gt_cellnbs, tracker_cellnbs = dataset_error_alpha[0]
                        error_values = dataset_error_alpha[1]

                        for index, (gt_cellnb, tracker_cellnb) in enumerate(zip(gt_cellnbs,tracker_cellnbs)):

                            framenb_start, framenb_end = man_track_pred[tracker_cellnb-1][1:3]
                            framenbs = np.arange(framenb_start,framenb_end+1)

                            for framenb in framenbs:

                                instance = cv2.imread(str(filepaths[0].parent / ('mask' + f'{framenb:03d}.tif')),cv2.IMREAD_ANYDEPTH)

                                mask = instance == tracker_cellnb
                                mask_color = np.zeros((mask.shape + (3,)))
                                mask_color[mask] = color

                                cell_loc = np.where(mask)
                                cell_loc = [int(np.median(cell_loc[0])), int(np.median(cell_loc[1]))]

                                movie_hota_Ass_Re[framenb][mask] = movie_hota_Ass_Re[framenb][mask] * transparency + mask_color[mask] * (1-transparency) 

                                error_value = error_values[tracker_cellnbs == tracker_cellnb].mean()

                                movie_hota_Ass_Re[framenb] = cv2.putText(
                                    movie_hota_Ass_Re[framenb],
                                    text = str(round(error_value,2)), 
                                    org=(cell_loc[1]-10,cell_loc[0]), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale = fontscale,
                                    color = (255,255,255),
                                    thickness=1,
                                    )

                            framenb_start, framenb_end = man_track_gt[gt_cellnb-1][1:3]
                            framenbs = np.arange(framenb_start,framenb_end+1)

                            for framenb in framenbs:

                                instance = cv2.imread(str(gt_path/ (dataset_id + '_GT') / 'TRA' / (f'man_track{framenb:03d}.tif')),cv2.IMREAD_ANYDEPTH)

                                mask = instance == gt_cellnb
                                mask_color = np.zeros((mask.shape + (3,)))
                                mask_color[mask] = color

                                cell_loc = np.where(mask)
                                cell_loc = [int(np.median(cell_loc[0])), int(np.median(cell_loc[1]))]

                                movie_hota_Ass_Pr[framenb][mask] = movie_hota_Ass_Pr[framenb][mask] * transparency + mask_color[mask] * (1-transparency) 

                                error_value = error_values[gt_cellnbs == gt_cellnb].mean()

                                movie_hota_Ass_Pr[framenb] = cv2.putText(
                                    movie_hota_Ass_Pr[framenb],
                                    text = str(round(error_value,2)), 
                                    org=(cell_loc[1]-10,cell_loc[0]), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale = fontscale,
                                    color = (255,255,255),
                                    thickness=1,
                                    )                        

                else:

                    for framenb in range(len(dataset_error)):

                        if len(dataset_error[framenb][alpha]) > 0:

                            cellnbs = np.array(dataset_error[framenb][alpha])

                            for cellnb in cellnbs:

                                if 'FN' in error:
                                    instance = cv2.imread(str(gt_path / (dataset_id + '_GT') / 'TRA' / f'man_track{framenb:03d}.tif'),cv2.IMREAD_ANYDEPTH)
                                    assert cellnb in instance

                                elif 'FP' in error:
                                    instance = cv2.imread(str(filepaths[framenb]),cv2.IMREAD_ANYDEPTH)
                                    assert cellnb in instance

                                mask = instance == cellnb
                                mask_color = np.zeros((mask.shape + (3,)))
                                mask_color[mask] = color

                                if 'FN' in error:
                                    if 'Div' in error:
                                        movie_hota_Div_FN[framenb][mask] = movie_hota_Div_FN[framenb][mask] * transparency + mask_color[mask] * (1-transparency) 
                                    else:
                                        movie_hota_Det_FN[framenb][mask] = movie_hota_Det_FN[framenb][mask] * transparency + mask_color[mask] * (1-transparency) 

                                elif 'FP' in error:
                                    if 'Div' in error:
                                        movie_hota_Div_FP[framenb][mask] = movie_hota_Div_FP[framenb][mask] * transparency + mask_color[mask] * (1-transparency) 
                                    else:
                                        movie_hota_Det_FP[framenb][mask] = movie_hota_Det_FP[framenb][mask] * transparency + mask_color[mask] * (1-transparency)

                                else:
                                    raise NotImplemented

            hota_movies = []

            for hota_movie,text in zip([movie_hota_Det_FN,movie_hota_Det_FP,movie_hota_Div_FN,movie_hota_Div_FP,movie_hota_Ass_Pr,movie_hota_Ass_Re],[['Det','FN'],['Det','FP'],['Div','FN'],['Div','FP'],['Ass','Pr'],['Ass','Re']]):

                hota_movie = np.concatenate((np.zeros((hota_movie.shape[0],60,hota_movie.shape[2],3)),hota_movie),axis=1)
               
                for img_hota in hota_movie:
                    img_hota = cv2.putText(
                        img_hota,
                        text = 'HOTA', 
                        org=(0,30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = (255,255,255),
                        thickness=1,
                        )
                        
                    img_hota = cv2.putText(
                        img_hota,
                        text = text[0], 
                        org=(0,42), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = (255,255,255),
                        thickness=1,
                        )

                    img_hota = cv2.putText(
                        img_hota,
                        text = text[1], 
                        org=(0,54), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = (255,255,255),
                        thickness=1,
                        )

                            
                hota_movies.append(hota_movie)

            hota_movies[0] = cv2.putText(
                hota_movies[0],
                text = 'alpha: ' + str(alpha_values[alpha]), 
                org=(0,10), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale,
                color = (255,255,255),
                thickness=1,
                )
            
            movies.append(spacer)
            movies.extend(hota_movies)

    movie = np.concatenate(movies,2)

    if display_FP_FN_pixels:
        movie_pixel_FN_FP = movie_bf.copy()
        movie_pixel_FN_FP = np.concatenate((np.zeros((movie_pixel_FN_FP.shape[0],60,movie_pixel_FN_FP.shape[2],3)),movie_pixel_FN_FP),axis=1)

        for idx, (filepath_gt, filepath_pred) in enumerate(zip(filepaths_gt,filepaths_pred)):

            instance_gt = (cv2.imread(str(filepath_gt),cv2.IMREAD_ANYDEPTH) > 0).astype(float)
            instance_pred = (cv2.imread(str(filepath_pred),cv2.IMREAD_ANYDEPTH) > 0).astype(float)

            FN = instance_gt - instance_pred
            FP = instance_pred - instance_gt

            FN = np.concatenate((np.zeros((60,FN.shape[1])),FN),axis=0)
            FP = np.concatenate((np.zeros((60,FP.shape[1])),FP),axis=0)

            movie_pixel_FN_FP[idx,FN > 0] = (255,0,0)
            movie_pixel_FN_FP[idx,FP > 0] = (0,0,255)

            movie_pixel_FN_FP[idx] = cv2.putText(
                movie_pixel_FN_FP[idx],
                text = 'FN' ,
                org=(0,54), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale+0.1,
                color = (255,0,0),
                thickness=1,
                )

            movie_pixel_FN_FP[idx] = cv2.putText(
                movie_pixel_FN_FP[idx],
                text = 'FP',
                org=(movie_pixel_FN_FP.shape[2]//2,54), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale+0.1,
                color = (0,0,255),
                thickness=1,
                )
    
        movie = np.concatenate((movie[:,:,:img.shape[1]*3],movie_pixel_FN_FP,movie[:,:,img.shape[1]*3:]),axis=2)

    for framenb,mov in enumerate(movie):
        mov = cv2.putText(
            mov,
            text = 'Frame: ' + str(framenb), 
            org=(100,15), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = fontscale+.1,
            color = (255,255,255),
            thickness=1,
            )

        mov = cv2.putText(
            mov,
            text = 'Dataset: ' + dataset_id, 
            org=(0,15), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = fontscale+0.1,
            color = (255,255,255),
            thickness=1,
            )

    height, width, _ = movie[0].shape
    crf=20
    verbose = 1
    filename = save_path / dataset_id / (f'pred_gt{flex_div}{"_ctc" if display_ctc_errors else ""}{"_hota" if display_hota_errors else ""}{"_FP_FN_pixels" if display_FP_FN_pixels else ""}_movie.mp4')
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


    mov = movie[:,:,:img.shape[1]*2]
    height, width, _ = mov[0].shape
    crf=20
    verbose = 1
    filename = save_path / dataset_id / ('pred_gt' + flex_div + '_movie_pred_gt_only.mp4')
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
    for frame in mov:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()

    mov = movie[:,:,:img.shape[1]*3]
    height, width, _ = mov[0].shape
    crf=20
    verbose = 1
    filename = save_path / dataset_id / ('pred_gt' + flex_div + '_movie_pred_gt_pixel_FP_FN_only.mp4')
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
    for frame in mov:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()