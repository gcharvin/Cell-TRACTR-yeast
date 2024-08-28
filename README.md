# Cell-TRACTR
This repository provides the implementation of [Cell-TRACTR](https://www.biorxiv.org/content/10.1101/2024.07.11.603075v1) paper by Owen O'Connor and Mary Dunlop. The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Trackformer](https://github.com/timmeinhardt/trackformer)

## Abstract


## Installation

Clone and enter this repository:

```git clone https://gitlab.com/dunloplab/Cell-TRACTR.git```

```cd Cell-TRACTR```

Install packages for Python 3.10:

```pip3 install -r requirements.txt```

Install ffmpeg

```conda install -c conda-forge ffmpeg```

Install PyTorch 1.13.1 and cuda 11.6 from [here](https://pytorch.org/get-started/previous-versions/#v1131).

Install pycocotools (with fixed ignore flag): ```pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'```

Install MultiScaleDeformableAttention package: ```python src/Cell-TRACTR/models/ops/setup.py build --build-base=src/Cell-TRACTR/models/ops/ install```

## Formatting Datasets

The mother machine dataset (moma) used in this paper can be found [here](https://zenodo.org/records/11237127).

The DeepCell dataset (DynamicNuclearNet-tracking-v1_0) can be found [here](https://datasets.deepcell.org/).

Both of these dataset are formatted the same as the Cell Tracking Challenge (CTC). More details on CTC formatting can be found [here](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf)

You may use your own custom dataset. Ensure your dataset is formatted in the CTC format. Your directory should have the same structure below with a train and val folder and optional test folder:

```data/moma/CTC/train```
```data/moma/CTC/val```
```data/moma/CTC/test```

You need a train and val folder for training. A test folder may be used for inference.

Then you need to use the ```create_coco_dataset_from_CTC.py``` script to convert the CTC formatted dataset into the COCO format. Since Cell-TRACTR processes multiple frames at once, videos with less than will not be included in the dataset used to train Cell-TRACTR. A COCO folder will be created with the structure shown below:

```data/moma/COCO/annotations```
```data/moma/COCO/man_track```
```data/moma/COCO/train```
```data/moma/COCO/val```


## Train Cell-TRACTR

You can train the model using the train.py script located in the src directory. There are a few ways to run the script, depending on the dataset you want to use:

1. Default Mode 

```python Cell-TRACTR/src/train.py```

If you don't specify a dataset, the script will automatically look for YAML configuration files in the cfgs folder. It will pick the first available configuration file and use that dataset for training.

2. Specifying a Dataset: You can also explicitly specify a dataset using the dataset argument. For example:

```python Cell-TRACTR/src/train.py with dataset='moma'```

When you specify a dataset, the script looks for a corresponding YAML file in the cfgs folder (e.g., train_moma.yaml or train_DynamicNuclearNet-tracking-v1_0.yaml). Ensure that the dataset name you provide matches the available configuration files.

Additional Notes:
- The path to the dataset should be correctly set in the corresponding YAML configuration file.
- The script automatically checks if a checkpoint.pth file exists in the results directory for the dataset. If it does, the script loads the existing configuration; otherwise, it creates a new one based on the specified dataset. This is used to continue training a model.

# Running inference for Cell-TRACTR

To run inference using the trained model, you can execute the pipeline.py script located in the src directory. The script can be run in several ways, depending on the dataset and configuration you want to use:

1. Default Inference

```python Cell-TRACTR/src/pipeline.py```

If you don't specify a dataset, the script will look for YAML configuration files in the cfgs folder and use the first available configuration file by default. It will then perform inference using the model trained on that dataset.

2. Specifying a Dataset for Inference: You can also explicitly specify a dataset using the dataset argument. For example:

```python Cell-TRACTR/src/pipeline.py with dataset='moma'```

The script will look for a corresponding YAML configuration file (e.g., pipeline_moma.yaml or pipeline_DynamicNuclearNet-tracking-v1_0.yaml) in the cfgs folder and use it for the inference process.

Additional Notes:
- The script initializes the model in evaluation mode and processes all the relevant image sequences (TIF files) in the dataset's test folder (/CTC/test).
- The script computes the frames per second (FPS) during inference and writes this information to an FPS.txt file in the output directory.
- If your dataset is not supported by the current implementation (e.g., lacking tracking or masks), an error will be raised.

## Visual

A time-lapse microscopy video of bacteria growing in the mother machine analyzed by Cell-TRACTR. The raw images used to generate this movie was taken from the test set within the [mother machine dataset](https://zenodo.org/records/11237127)

![[Video]](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2VrOW16djJnYzJ4cWhsd2F0cjNtNzVnazgzNjZuMjhucmdoNGkwYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cwNDVhxqTPYMYMxd75/giphy.gif)

A time-lapse microscopy video of mammalian cells growing on 96-well plates.  The raw images used to generate this movie was taken from the test set within the DeepCell dataset - [DynamicNuclearNet Tracking](https://datasets.deepcell.org/data)

![Video](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjFzbXBkZmFpYnZsNjdpbmlvZjY1cGFpdGc0NnNuZWoyOHg4bWN3YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/fhTdHoCSARZjrRpyyn/giphy.gif)

## License

Cell-TRACTR is released under the [Apache 2.0 License](LICENSE).