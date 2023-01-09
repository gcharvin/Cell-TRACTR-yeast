from pathlib import Path  
import re
import shutil
from tqdm import tqdm

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/raw_data')

fps = sorted(list((datapath / 'inputs').glob('*.png')))

for folder in ['inputs','outputs','img','previmg']:
    (datapath.parent / 'new_dataset' / 'raw_data' / folder).mkdir(exist_ok=True)

for fp in tqdm(fps):
    fn = fp.name
    numbers = re.findall('\d+',fn)    
    chars = re.findall('\D+',fn)

    if len(numbers) == 7:
        replace = numbers[0] + chars[0] + numbers[1] + chars[1] + numbers[2] #+ chars[2] + numbers[3]

        fn = fn.replace(replace,'Exp01')
        fn = fn.replace('ROI','Chamber')
    else:

        fn = 'Exp02_TrainingSet1_' + fn
        fn = fn.replace(f'Chamber{numbers[-2]}',f'Chamber{int(numbers[-2]):06d}')        

    for folder in ['inputs','outputs','img','previmg']:
        shutil.copy(fp.parents[1] / folder / fp.name,fp.parents[2] / 'new_dataset' / 'raw_data' / folder / fn)

