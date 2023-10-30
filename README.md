# VGGIshIsh

This is an implementation of the VGGIshIsh model, proposed in [Taming Visually Guided Audio Generation](https://arxiv.org/abs/2110.08791) by Vladimir Iashin and Esa Rahtu. The code is following closely the official implementation [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN). In this repo, the model is used to classify between hit an scratch sounds from the [Greatest Hit dataset](https://andrewowens.com/vis/).

## Installation

Install the conda environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Data and Preprocessing

To download data go to the official website of the Greates Hit-dataset https://andrewowens.com/vis/.https://andrewowens.com/vis/.

After downloading the data, preprocess it with the provided wav_to_melspec.py script:

```bash
python wav_to_melspec.py --data_path=path/to/data --save_path=path/to/save
```

## Usage

To train the model, run:

```bash
python train.py config=configs/vggishish.yaml
```

To test the model, run:

```bash
python test.py config=configs/vggishish.yaml ckpt_path=path/to/ckpt/file.pt
```

## Results

To view training results in TensorBoard, run:

```bash
tensorboard --logdir=logs
```
