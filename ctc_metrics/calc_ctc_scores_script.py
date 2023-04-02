from pathlib import Path
from eval_ctc import calc_ctc_scores
import pandas as pd
import re 

modelname = '230401_moma_track_two_stage_dn_enc_dn_track_dab_mask'
test_path = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results') / modelname / 'test'

# test_path = Path('/projectnb/dunlop/ooconnor/object_detection/delta/results/ctc')
# test_path = Path('/projectnb/dunlop/ooconnor/object_detection/embedtrack/results/test')

gt_dir = Path('/projectnb/dunlop/ooconnor/object_detection/data/moma/test')

series_list = []

res_paths = [res_path for res_path in sorted(test_path.iterdir()) if re.findall('\d\d$',res_path.name)]

for res_path in res_paths:
    
    gt_path = gt_dir / (res_path.name + '_GT')
    metrics = calc_ctc_scores(res_path, gt_path)

    series_list.append(pd.Series(metrics.values(), index=metrics.keys(), name=res_path.name))

df = pd.concat(series_list, axis=1)
df['AVG'] = df.mean(1)

df.to_csv(test_path / 'ctc_scores.csv')

