# MiniML

This repository contains some of my experiments and trials with Deep Learning models that do not fit well in their own repositories. They are implemented using [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to simplify training and remove boilerplate.

## Installation

Clone the repository using:

```bash
git clone https://github.com/williamcorsel/miniml
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Training, evaluating, and testing of the models is implemented using the Pytorch Lighting [CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). This can be used to easily train arbitrary models (implemented as ModelModules) on arbitrary datasets (implemented using DataModules). The basic recipe is as follows:

```bash
python miniml.py {fit,validate,test,predict} --model model_name --data data_name
```

It also allows you to set hyperparameters using the CLI or `yaml` files. Some example configurations are provided in the [configs](configs) folder. The can be used as follows:

```bash
python miniml.py {fit,validate,test,predict}  --config path_to_config.yaml
```

## Capabilities
### Image Classification

Contains simple versions of the [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) models to train on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The datasets and predictions of the models can be visualised using [FiftyOne](https://docs.voxel51.com/) by using:

```bash
python visualise_image_classification.py --ckpt_path path_to_checkpoint.ckpt --data_dir path_to_data --name fiftyone_dataset_name
```

### 3D Model Generation

Proof of concept of a 3D-GAN model to generate 3D Minecraft houses from the [Craft3D dataset](https://github.com/facebookresearch/voxelcnn)