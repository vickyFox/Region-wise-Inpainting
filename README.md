# Image inpainting via Region-Wise Convolution 
This repository implements the training and testing code for "Region-wise Generative Adversarial Image Inpainting for Large Missing Areas". We propose an generic inpainting framework capable of handling with incomplete images on both continuous and discontinuous large missing areas, in an adversarial manner. we extend upon our prior conference publication "Coarse-to-Fine Image Inpainting via Region-wise convolutions and Non-local correlation" that mainly focuses on the discontinuous missing areas using the region-wise convolutions, suffering severe artifacts when the large missing areas are continuous. Some results are as follows. 
![](https://github.com/vickyFox/Region-wise-Inpainting/blob/master/images/image1.png)
## RUN

**Requirements:**

The code was trained and tested with Tensorflow 1.x.0.CUDA 9.0 ,Python 3.6 

- Install python3.

- Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).

- clone this repo

  ```
  git clone https://github.com/vickyFox/Region-wise-Inpainting.git
  cd  Region-wise-Inpainting Private
  ```

**Datasets**

- **Face dataset**: 28000 training images and 2000 test images from [CelebA-HQ](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28)
- **Paris dataset**: 14900 training images and 100 test images from [Paris](https://github.com/pathak22/context-encoder)
- **natural images**: original training images and sampled test images(images sampled from the origin testing images) from [Places2](http://places2.csail.mit.edu/)

**Training**
- Download **[vgg16.npy](https://drive.google.com/open?id=1iYsD62btPTL5ZIirGJIMbIU03ujJRNEU)** and put it under ```/vgg```.
- Prepare training images.
- Run ```python train.py --train_data_path your_image_path --output your_output_path```
- Run ```python train.py -h``` to get more command line arguments

**Testing**

 ```python test.py --test_data_path your_data_path --mask_path your_mask_path --model_path pretrained_model```
- We provide the pretrained models incuding celebA-HQ, Paris and Places2.
- The default results will be saved under the results folder. You can set ```--file_out``` to choose a new path.

## Pretrained
Download the pre-trained models and put them under the default directory ```\models``` or any other path you want.
- continuous model: [CelebA-HQ](https://drive.google.com/open?id=1q7tuopiOwRPZOPYG5076EPoFLPo0CQF6) | [Paris](https://drive.google.com/open?id=1STSPPyLQ4LjWj-juT5nMXJ9X5h7N_fjO) | [Places2](https://drive.google.com/open?id=1zYxZPU7L6Ongu0tlkHEJTPxf9Cw4RvqQ)
- discontinuous model: [CelebA-HQ](https://drive.google.com/open?id=1IsRRRcGIg-I1Dklxz09wO7UDiB5Q3Map) | [Paris](https://drive.google.com/open?id=1vtT6jiya2tSo6QVzVpbMFwoz6H9oGeZc) | [Places2](https://drive.google.com/open?id=1b9DubT3WTIKQ_GcQNAXxlB68c3C5jKX0) 

Our main novelty of this project is the ability to handle with incomplete images on both continuous and discontinuous largemissing areas.

