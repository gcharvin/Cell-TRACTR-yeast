from pathlib import Path
from eval_ctc import calc_ctc_scores

modelname = '230203_prev_prev_track_final_two_stage_dn_enc_dn_track_dab_mask'

test_path = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results') / modelname / 'test'
img_dir = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/moma/test/ctc_data')

for res_path in sorted(test_path.iterdir()):
    
    img_path = img_dir / (res_path.name + '_GT')
    metrics = calc_ctc_scores(res_path, img_path)