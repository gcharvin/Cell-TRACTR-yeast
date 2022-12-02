from pathlib import Path  
import re
import math
import cv2
from tqdm import tqdm

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/trackformer_2d/data/cells/raw_data_ordered')
datapath = Path('/projectnb/dunlop/ooconnor/object_detection/trackformer_2d/data/cells')

folders = ['train','val']
remove = False
target_size = (256,32)

fps = sorted(list((datapath / 'combined').glob('*.png')))

for f,fp in enumerate(tqdm(fps)):

    if '_cur' in fp.name:
        assert (fp.parent / (fp.stem.replace('_cur','_prev') + '.png')).exists()
        continue
    elif '_prev' in fp.name:
        assert (fp.parent / (fp.stem.replace('_prev','_cur') + '.png')).exists()

    fn = fp.name
    inputs = cv2.imread(str(fp),cv2.IMREAD_UNCHANGED)      

    assert inputs.shape == target_size

    framenb = list(map(int,re.findall('\d+',fn)))[-1]
    pad = re.findall('\d+',fp.name)[-1].count('0') + int(math.log10(framenb))+1
    fn = fn.replace(re.findall('\d+', fn)[-1],f'{framenb-1:0{str(pad)}d}')
    fn = fn.replace('_prev','_cur')

    if (fp.parent / fn).exists():
        outputs = cv2.imread(str(fp.parent / fn),cv2.IMREAD_UNCHANGED)
        assert outputs.shape == target_size

        if not ((inputs > 0) == (outputs > 0)).all():
            print(fn,fp.name)
            if remove:
                #### TODO double check that this works as expected
                for folder in folders:
                    if (fp.parents[1] / folder / fn).exists():
                        (fp.parents[1] / folder / fn).unlink()
                    if (fp.parents[1] / folder / fp.name).exists():
                        (fp.parents[1] / folder / fp.name).unlink()
                if (fp.parent / fp.name).exists():
                    (fp.parent / fp.name).unlink()
                if (fp.parent / fn).exists():
                    (fp.parent / fn).unlink()
