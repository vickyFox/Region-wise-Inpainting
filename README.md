# Image inpainting via Region-Wise Convolution 
This repository implements the training and testing code for "Region-wise Generative Adversarial Image

Inpainting for Large Missing Areas" .We can restore semantically reasonable and visually realistic images from an incomplete input. Some results are as follows.
## RUN

**Requirements:**

The code was trained and tested with Tensorflow 1.x.0.CUDA 9.0 ,Python 3.6 

- Install python3.

- Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).

- clone this repo

  ```
  git clone
  cd
  ```

**Datasets**

- **Face dataset**: 28000 training images and 2000 test images from [CelebA-HQ](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28)
- **Paris dataset**: 14900 training images and 100 test images from [Paris](https://github.com/pathak22/context-encoder)
- **natural images**: original training images and sampled test images(images sampled from the origin testing images) from [Places2](http://places2.csail.mit.edu/)

**Training**
