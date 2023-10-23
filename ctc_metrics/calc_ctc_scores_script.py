from pathlib import Path
from eval_ctc import calc_ctc_scores
import pandas as pd
import re 
import time 

modelname = '230831_moma_flex_div_CoMOT_track_two_stage_dn_enc_dn_track_dn_track_group_dab_intermediate_mask_4_enc_4_dec_layers'
test_path = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results') / modelname / 'test' / 'CTC'

# test_path = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/test/CTC')
# test_path = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test/CTC')

gt_dir = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test/CTC')

series_list = []

start_time = time.time()

res_paths = [res_path for res_path in sorted(test_path.iterdir()) if re.findall('\d\d$',res_path.name)]
num_images = 0 
for res_path in res_paths:
    
    gt_path = gt_dir / (res_path.name + '_GT')
    num_images += len(list((gt_path / 'TRA').iterdir()))
    metrics = calc_ctc_scores(res_path, gt_path)

    series_list.append(pd.Series(metrics.values(), index=metrics.keys(), name=res_path.name))

df = pd.concat(series_list, axis=1)
df['AVG'] = df.mean(1)

df.to_csv(test_path / 'ctc_scores.csv')

print(f'There are {num_images} images in text set')

end_time = time.time()
diff = end_time - start_time

print(f'It took {round(diff,3)} seconds')