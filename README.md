# Predicting the Most Dominant Color of an Image

A deep learning project that trains a network to look at an image and output its single most dominant color as an RGB value. First project for the Deep Learning course; the full write-up (in Serbian) is in `Projekat_1_Duboko_učenje.pdf`.

## Approach

**Labelling.** There is no ready-made dataset for this, so the labels are computed. `dataset/label_dataset.py` counts every distinct RGB value in an image on the GPU with a custom CUDA kernel (PyCUDA) and writes the most frequent one to `dataset.csv`: about 14,000 images, each with its dominant color as RGB and hex.

**Analysis and cleaning.** The scripts in `data_analysis/` look at the label distribution: black-and-white images, vibrancy, unique colors, RGB clustering, and scanning artifacts. `preprocessing/` drops the images flagged as artifacts and produces `cleaned_dataset.csv`.

**Models.** Regression to three values (R, G, B scaled to 0 to 1), trained with MSE loss and Adam on a 60/20/20 split:

| Model | Description |
| --- | --- |
| `lenet_v1` | LeNet-style CNN trained from scratch |
| `resnet_v1` | Pretrained ResNet-18, only the final layer trained |
| `resnet_v2` | Pretrained ResNet-18 with the last block unfrozen, augmentation and a learning-rate scheduler |

## Results

A prediction counts as correct when all three channels are within 0.2 of the label. Validation accuracy by epoch:

| Model | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 15 | Epoch 20 |
| --- | --- | --- | --- | --- | --- |
| LeNet v1 | 5.18% | 3.57% | 9.56% | 13.02% | 13.06% |
| ResNet v1 | 28.42% | 33.18% | 32.80% | 44.56% | 39.81% |
| ResNet v2 | 30.28% | 44.52% | 52.61% | 55.27% | 57.33% |

Transfer learning matters far more than architecture size here, and unfreezing the last ResNet block adds about 17 points over training the head alone.

## Layout

```
dataset/          labelling scripts and the label CSVs
data_analysis/    exploration of the label distribution
preprocessing/    artifact removal, cleaned labels
models/           training scripts and trained weights (.pth)
compare_models/   accuracy and architecture comparison plots
```

## Running it

The images themselves are not in the repo, only the labels. The training scripts have absolute Windows paths at the top of `__main__`: point `csv_path` and `image_dir` at your own copies before running.

```bash
pip install torch torchvision pandas pillow tqdm matplotlib
python models/resnet_v2.py
```

Labelling from scratch also needs an NVIDIA GPU with PyCUDA and OpenCV.
