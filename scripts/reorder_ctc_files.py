from pathlib import Path
import re
import os

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/data/2D/CTC')

folderpaths = sorted(list(datapath.iterdir()))

total_num = len(folderpaths) // 2

for f, folderpath in enumerate(folderpaths):

    num = f // 2 + 1
    fol = f'{num:02d}'

    if bool(re.search('\d\d$',folderpath.name)) and folderpath.name != fol:
        os.rename(folderpath,folderpath.parent / fol)
        print(folderpath.name)
    elif bool(re.search('\d\d_GT',folderpath.name)) and folderpath.name != (fol + '_GT'):
        os.rename(folderpath,folderpath.parent / (fol + '_GT'))
        print(folderpath.name)
