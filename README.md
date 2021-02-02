# Zero-Shot-Super-Resolution
This repository contains a PyTorch implementation of Zero Shot Super Resolution using Residual Feature Fusion classifier and ECA module.<br/>

I have restructured the classifier. In each step of the classifier two convolutional features are taken parallelly and they are applied convolving kernels of size 3x3 and 5x5 respectively and then the features are concatenated together. This same process is done another time. Then ECA layer is used on the concatenated features. To learn the residuals low resolution image is added with the output of the eca layer. This whole block is applied twice in the network before the final output.<br/>

## Results
<img src="images/bicubic.png"/> <img src="images/glasner.png"/> <img src="images/imgSR.png"/> <img src="images/imgOG.png"/><br/>
Fig1. (a)Bicubic (b)Glasner (c)This Model (d)Original<br/>

The results of this model are:<br/>

PSNR: 34.47285328485344<br/>
SSIM: 0.9725940096716498

## Acknowledgement
ZSSR paper: https://arxiv.org/pdf/1712.06087.pdf<br/>
ECA-Net paper: https://arxiv.org/pdf/1910.03151.pdf<br/>

Part of this code is taken from [MayankSingal](https://github.com/MayankSingal/PyTorch-Zero-Shot-Super-Resolution)



