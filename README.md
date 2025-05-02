# dl4med-chestxray
Deep Learning Model Comparison for NIH Chest X-Ray Dataset

This repository contains code and results for a final project that implements and compares multiple deep learning models for multi-label classification of chest pathologies using the NIH Chest X-ray14 dataset available via Box (https://nihcc.app.box.com/v/ChestXray-NIHCC). The goal is to assess and compare the performance of different architectures on detecting 14 common thoracic diseases from chest radiographs.

Here, we tested three deep learning architectures:
 - CNN (Baseline Model)
 - Pre-trained ResNet50
 - Pre-trained BEiT (BERT Pretraining of Vision Transformer)

# References
Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). 
ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
https://arxiv.org/abs/1512.03385

Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT Pre-Training of Image Transformers. International Conference on Learning Representations (ICLR), 2022.
https://arxiv.org/abs/2106.08254
