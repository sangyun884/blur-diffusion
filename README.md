# blur-diffusion
This is the codebase for our paper Progressive Deblurring of Diffusion Models for Coarse-to-Fine Image Synthesis.

![Teaser image](./images/main_fig.jpg)

> **Progressive Deblurring of Diffusion Models for Coarse-to-Fine Image Synthesis**<br>
> Sangyun Lee<sup>1</sup>, Hyungjin Chung<sup>2</sup>, Jaehyeon Kim<sup>3</sup>, ‪Jong Chul Ye<sup>2</sup>

> <sup>1</sup>Soongsil University, <sup>2</sup>KAIST, <sup>3</sup>Kakao Enterprise<br>

> Paper: https://arxiv.org/abs/2207.11192<br>

> **Abstract:** *Recently, diffusion models have shown remarkable results in image synthesis by gradually removing noise and amplifying signals. 
Although the simple generative process surprisingly works well, is this the best way to generate image data? For instance, despite the fact that human perception is more sensitive to the low-frequencies of an image, diffusion models themselves do not consider any relative importance of each frequency component. Therefore, to incorporate the inductive bias for image data, we propose a novel generative process that synthesizes images in a coarse-to-fine manner. First, we generalize the standard diffusion models by enabling diffusion in a rotated coordinate system with different velocities for each component of the vector. We further propose a blur diffusion as a special case, where each frequency component of an image is diffused at different speeds. Specifically, the proposed blur diffusion consists of a forward process that blurs an image and adds noise gradually, after which  a corresponding reverse process deblurs an image and removes noise progressively. Experiments show that proposed model outperforms the previous method in FID on LSUN bedroom and church datasets.*

## Train
```bash train.sh```
## Visualization
```bash eval_x0hat.sh```
## Dataset
Image files are required to compute FID during training.
