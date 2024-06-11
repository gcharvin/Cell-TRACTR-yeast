# XXX
This repository provides the implementation of [XXX](link to paper) paper by Owen O'Connor and Mary Dunlop. The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Trackformer](https://github.com/timmeinhardt/trackformer)

## Abstract


## Installation

Clone and enter this repository:

```git clone https://gitlab.com/dunloplab/deeplearning/cell-trackformer.git```

```cd cell-trackformer```

Install packages for Python 3.10:

```pip3 install -r requirements.txt```

Install ffmpeg

```conda install -c conda-forge ffmpeg```

Install PyTorch 1.13.1 and cuda 11.6 from [here](https://pytorch.org/get-started/previous-versions/#v1131).

Install pycocotools (with fixed ignore flag): ```pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'```

Install MultiScaleDeformableAttention package: ```python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install```

## Train XXX

XXX/src/train.py

You need to update 2 variables

    dataset: str - name of the dataset
    respath: Pathlib path - path of the results directory

## Visual

A time-lapse microscopy video of bacteria growing in the mother machine analyzed by XXX. The raw images used to generate this movie was taken from the test set within the [mother machine dataset](https://zenodo.org/records/11237127)

![[Video]](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2VrOW16djJnYzJ4cWhsd2F0cjNtNzVnazgzNjZuMjhucmdoNGkwYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cwNDVhxqTPYMYMxd75/giphy.gif)

A time-lapse microscopy video of mammalian cells growing on well plates.  The raw images used to generate this movie was taken from the test set within the DeepCell dataset - [DynamicNuclearNet Tracking](https://datasets.deepcell.org/data)

![Video](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjFzbXBkZmFpYnZsNjdpbmlvZjY1cGFpdGc0NnNuZWoyOHg4bWN3YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/fhTdHoCSARZjrRpyyn/giphy.gif)

## License
For open source projects, say how it is licensed.
