# Recurrent Residual CNNs based on U-Net (R2U-Net) for Medical Image Segmentation 

**Jacqueline Jia (jjia6), Faizaan Vidhani (fvidhani), Josh Woo (jwoo15)**
 
**Introduction**

Recent developments in computer-aided diagnostics and deep convolutional neural networks have allowed for an enhanced ability to classify, segment, detect, and track medical images. Such technologies have enabled more accurate and time/cost efficient diagnoses within clinical settings. In our project we implement a model that performs image segmentation on medical images of skin lesions, each of which is provided a corresponding label. This model will integrate three deep learning techniques to address this supervised segmentation problem; U-Net encoding/decoding, recurrence, and residual networks. While the U-Net architecture is typically sufficient for general image segmentation (compressing and then decompressing the image through convolution), the original paper justified the additional modifications by observing a need for improved training accuracy and better feature representation. We ultimately chose this topic because the referenced paper provided an opportunity to integrate several of the concepts which we have learned this semester into a single model that produced impressive images. 

**Methodology**


<img width="626" style="display: block; margin: 0 auto" alt="Methodology Image" src="https://user-images.githubusercontent.com/39887209/211691371-d582ed30-1f7e-4c33-9793-fead40d784d0.png">
