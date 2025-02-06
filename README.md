# sam2



Usage: torchrun --nproc-per-node=6 train.py --batch_size=6 --encresume="safire_encoder_pretrained.pth" --resume="" --num_epochs=150




TODO: complete the dataset 1task 33day
TODO: complete method 33day
TODO: start training

TODO: complete the dataset 2task 32day


TODO: design experiments




我其实需要一周之内close所有任务，然后开始写论文











主要是写个代码，需要根据CAT-Net的划分，把数据集划分成train/test/val

Complete Dataset：
tampCOCO: tampCOCO是一个用于篡改检测和定位任务的图像数据集，它由COCO数据集中的图像及其对应的伪造篡改区域和掩码组成。

@article{kwon2022learning,
  title={Learning jpeg compression artifacts for image manipulation detection and localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume={130},
  number={8},
  pages={1875--1895},
  year={2022},
  publisher={Springer}
}


Casiav2:



Training Stream：
1st phase: 
traditional dataset, CAT-Protocal datasets

2nd phase: 
SD 15（generative）

3rd phase: 
SD 3


4th phase: 



5th phase: 





TGIF

Kolors







Testset：


FantasticReality

COVERAGE: 特点是原始图像已经包含相似物体，增加了检测难度，伪造区域较大。
https://github.com/wenbihan/coverage

@inproceedings{wen2016,
  author={Wen, Bihan and Zhu, Ye and Subramanian, Ramanathan and Ng, Tian-Tsong and Shen, Xuanjing and Winkler, Stefan},
  title={COVERAGE – A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION},
  year={2016},
  booktitle={IEEE International Conference on Image processing (ICIP)},
  pages={161--165}
}

IMD2020: 手动选择区域，并使用OpenCV和机器学习模型进行修复，但部分图像不真实，且伪造图像进行过一些额外的更改。

https://staff.utia.cas.cz/novozada/db/

@INPROCEEDINGS{Novozamsky_2020_WACV,
author = {Novozamsky, Adam and Mahdian, Babak and Saic, Stanislav},
title = {IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting Manipulated Images},
booktitle = {2020 IEEE Winter Applications of Computer Vision Workshops (WACVW)},
year = {2020},
month = {March},
pages = {71-80}
}


DSO-1: 
https://recodbr.wordpress.com/code-n-data/#dso1_dsi1


Casiav1:
https://github.com/namtpham/casia1groundtruth

@inproceedings{Dong2013,
doi = {10.1109/chinasip.2013.6625374},
url = {https://doi.org/10.1109/chinasip.2013.6625374},
year = {2013},
month = jul,
publisher = {{IEEE}},
author = {Jing Dong and Wei Wang and Tieniu Tan},
title = {{CASIA} Image Tampering Detection Evaluation Database},
booktitle = {2013 {IEEE} China Summit and International Conference on Signal and Information Processing}
}


CocoGlide:

https://github.com/grip-unina/TruFor

@InProceedings{Guillaro_2023_CVPR,
   author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
   title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month     = {June},
   year      = {2023},
   pages     = {20606-20615}
}


Columbia: The Columbia Uncompressed Image Splicing Detection Evaluation Dataset is a collection of 183 authentic and 180 spliced uncompressed, high-resolution images, taken with four different cameras, designed for evaluating image splicing detection algorithms and includes detailed EXIF data and edgemasks for ground truth.

https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/

@inproceedings{hsu06crfcheck,
author = {Y.-F. Hsu and S.-F. Chang},
title = {Detecting Image Splicing Using Geometry Invariants and Camera Characteristics Consistency},
booktitle = {International Conference on Multimedia and Expo},
year = {2006},
location = {Toronto, Canada}
}




TGIF, a collection of approximately 75,000 forged images created using popular text-guided inpainting methods like SD2, SDXL, and Adobe Firefly.


AutoSplice 是一个用于媒体取证的文本提示操控图像数据集，包含5894张由DALL-E2模型生成的、通过文本提示引导的局部或全局篡改图像和真实图像，以及三种不同质量的JPEG压缩版本及其对应的篡改掩码和标题。

@inproceedings{jia2023autosplice,
  title={Autosplice: A text-prompt manipulated image dataset for media forensics},
  author={Jia, Shan and Huang, Mingzhen and Zhou, Zhou and Ju, Yan and Cai, Jialing and Lyu, Siwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={893--903},
  year={2023}
}




CocoGlide








SafireMS  是一个用于图像篡改检测的多源伪造图像数据集，包含自动生成的 SafireMS-Auto（约123k张图像）和专家手工制作的 SafireMS-Expert（238张图像，包含2-4个源区域），旨在推动多源图像篡改定位的研究。
SafireMS-Auto: 包含约123k张自动生成的伪造图像的大规模数据集，用于预训练。包含四种伪造类型：复制-移动 (CM)、拼接 (SP)、生成式重建 (GN) 和 基于AI的移除修复 (RM)。
SafireMS-Expert: 包含238张由专家手动创建的、具有2到4个源区域的多源数据集，用于评估多源分割性能。包含多种伪造类型组合。
@article{kwon2024safire,
  title={SAFIRE: Segment Any Forged Image Region},
  author={Kwon, Myung-Joon and Lee, Wonjun and Nam, Seung-Hun and Son, Minji and Kim, Changick},
  journal={arXiv preprint arXiv:2412.08197},
  year={2024}
}

Dolos 构建了一个包含超过125,000张真实图像、完全合成图像和局部篡改图像的数据集，主要针对由扩散模型生成的深度伪造人脸图像，用于研究和评估弱监督条件下的深度伪造定位算法。


| Dataset Type | Source | Training | Validation | Testing | Total |
|--------------|---------|-----------|------------|---------|--------|
| **Real Images** |
| CelebA-HQ | - | 9,000 | 900 | 900 | 10,800 |
| FFHQ | - | 9,000 | 900 | - | 9,900 |
| **Fully Synthetic (P2 Model)** |
| Based on CelebA-HQ | P2 | 9,000 | 1,000 | - | 10,000 |
| Based on FFHQ | P2 | 9,000 | 1,000 | - | 10,000 |
| **Partially Manipulated** |
| Repaint–P2/CelebA-HQ | P2 | 30,000 | 3,000 | 8,500 | 41,500 |
| Repaint–P2/FFHQ | P2 | 30,000 | 3,000 | - | 33,000 |
| Repaint–LDM/CelebA-HQ | LDM | 9,000 | 900 | 900 | 10,800 |
| LaMa/CelebA-HQ | LaMa | 9,000 | 900 | 900 | 10,800 |
| Pluralistic/CelebA-HQ | Pluralistic | 9,000 | 900 | 900 | 10,800 |
| **Main Test Sets** |
| Localization | Repaint–P2/CelebA-HQ | - | - | 8,500 | 8,500 |
| Detection | CelebA-HQ (Real) + Repaint–P2/CelebA-HQ | - | - | 1,800 | 1,800 |



@inproceedings{țanțaru2024weakly,
  title={Weakly-supervised deepfake localization in diffusion-generated images},
  author={Ț{\^a}nțaru, Dragoș-Constantin and Oneaț{\u{a}}, Elisabeta and Oneaț{\u{a}}, Dan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6258--6268},
  year={2024}
}











