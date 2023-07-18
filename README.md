##  [Physical Model and Image Translation Fused Network for Single-Image Dehazing](PR 2023)
 Official implementation.

---

by Yan Zhao Su, Chuan He et al. 

### Citation
https://doi.org/10.1016/j.patcog.2023.109700  
https://www.sciencedirect.com/science/article/pii/S0031320323003989  
[
Yan Zhao Su, Chuan He, Zhi Gao Cui, Ai Hua Li, Nian Wang,
Physical model and image translation fused network for single-image dehazing,
Pattern Recognition,
Volume 142,
2023,
109700
]
###Abstract
The visibility and contrast of images captured in adverse weather such as haze or fog degrade dramatically, which further hinders the accomplishment of high-level computer vision tasks such as object detection and semantic segmentation in these conditions. Many methods have been proposed to solve image dehazing problem by using image translation networks or physical model embedding in CNNs. However, the physical model cannot effectively describe the hazy generation process in complex scenes and estimating the model parameters with only a hazy image is an ill-posed problem. Image translation-based methods may lead to artefacts or colour shifts in the recovered results without the guidance or constraints of physical model information. In this paper, an end-to-end physical model and image translation fused network is proposed to generate realistic haze-free images. Since the transmission map can express the haze distribution in the scene, the proposed method adopts an encoder with a multiscale residual block to extract hazy image features, and two separate decoders to recover a clear image and to estimate the transmission map. The multiscale features of the transmission map and image translation are fused to guide the decode processes with a conditional attention feature fusion block, which is composed of sequential channelwise and spatialwise attention. Moreover, a multitask and multiscale deep supervision mechanism is adopted to enhance the feature fusion and recover more image details. The algorithm can efficiently fuse the physical model information and the hazy image translation to address the problem existent in the methods only based on physical model embedding or direct image translation. Experimental results on the visual quality enhancement of hazy images and semantic segmentation tasks in hazy scenes demonstrate that our model can efficiently recover haze-free images, while performing on par with state-of-the-art methods.  
### Dependencies and Installation

* python3
* PyTorch>=1.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)


#### Test
The code has been tested on the enviroment: Windows X64, pytorch1.8.0+cu101.  
Download the files, and the pre-trained models are in the 'model' file.  
Change the test directory in the test_reside.py or test_foggy_city.py with the model trained on ITS or Foggy-Cityscapes
 ```shell
 python test_reside.py
```