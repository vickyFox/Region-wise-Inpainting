# Region-wise Generative Adversarial Image Inpainting for Large Missing Areas
This repository implements the training and testing code for "Region-wise Generative Adversarial Image Inpainting for Large Missing Areas". We propose an generic inpainting framework capable of handling with incomplete images on both continuous and discontinuous large missing areas, in an adversarial manner. We extend upon our prior conference publication "[Coarse-to-Fine Image Inpainting via Region-wise convolutions and Non-local correlation](https://www.ijcai.org/proceedings/2019/0433.pdf)" that mainly focuses on the discontinuous missing areas using the region-wise convolutions, suffering severe artifacts when the large missing areas are continuous. Some results are as follows. 

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
- Pretrained model: [CelebA-HQ](https://drive.google.com/open?id=1q7tuopiOwRPZOPYG5076EPoFLPo0CQF6) | [Paris](https://drive.google.com/open?id=1STSPPyLQ4LjWj-juT5nMXJ9X5h7N_fjO) | [Places2](https://drive.google.com/open?id=1zYxZPU7L6Ongu0tlkHEJTPxf9Cw4RvqQ)
- ijcai Pretrained model: [CelebA-HQ](https://drive.google.com/open?id=1IsRRRcGIg-I1Dklxz09wO7UDiB5Q3Map) | [Paris](https://drive.google.com/open?id=1vtT6jiya2tSo6QVzVpbMFwoz6H9oGeZc) | [Places2](https://drive.google.com/open?id=1b9DubT3WTIKQ_GcQNAXxlB68c3C5jKX0)

Our main novelty of this project is the ability to handle with incomplete images on both continuous and discontinuous large missing areas.

## Citing
```
@ARTICLE{2019arXiv190912507M,
       author = {{Ma}, Yuqing and {Liu}, Xianglong and {Bai}, Shihao and {Wang}, Lei and
         {Liu}, Aishan and {Tao}, Dacheng and {Hancock}, Edwin},
        title = "{Region-wise Generative Adversarial ImageInpainting for Large Missing Areas}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2019",
        month = "Sep",
          eid = {arXiv:1909.12507},
        pages = {arXiv:1909.12507},
archivePrefix = {arXiv},
       eprint = {1909.12507},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190912507M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@inproceedings{ma2019inpainting,
  title={Coarse-to-Fine Image Inpainting via Region-wise Convolutions and Non-Local Correlation.},
  author={Yuqing Ma and Xianglong Liu and Shihao Bai and Lei Wang and Dailan He and Aishan Liu},
  booktitle={IJCAI},
  year={2019}
}
```
