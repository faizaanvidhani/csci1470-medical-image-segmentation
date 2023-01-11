# Skin Cancer Lesions Image Segmentation using Recurrent Residual CNNs based on U-Net (R2U-Net)

**Jacqueline Jia (jjia6), Faizaan Vidhani (fvidhani), Josh Woo (jwoo15)**

<img width="753" alt="Medical Image Segmentation Poster" src="https://user-images.githubusercontent.com/39887209/211692660-f95d9cd7-aac3-429c-9898-67f238d30f92.png">
 
**Introduction**

Recent developments in computer-aided diagnostics and deep convolutional neural networks have allowed for an enhanced ability to classify, segment, detect, and track medical images. Such technologies have enabled more accurate and time/cost efficient diagnoses within clinical settings. In our project we implement a model that performs image segmentation on medical images of skin lesions, each of which is provided a corresponding label. This model will integrate three deep learning techniques to address this supervised segmentation problem; U-Net encoding/decoding, recurrence, and residual networks. While the U-Net architecture is typically sufficient for general image segmentation (compressing and then decompressing the image through convolution), the original paper justified the additional modifications by observing a need for improved training accuracy and better feature representation. We ultimately chose this topic because the referenced paper provided an opportunity to integrate several of the concepts which we have learned this semester into a single model that produced impressive images. 

**Methodology**

<img width="707" alt="Screen Shot 2023-01-10 at 6 48 43 PM" src="https://user-images.githubusercontent.com/39887209/211692305-c18adc82-fe4e-4eab-94b4-c017189d2aaf.png">

We implemented a U-Net architecture with recurrent residual convolutional networks: this includes convolutional encoding and decoding units that take images as inputs and produce the segmentation feature maps with their respective pixel classes. The model architecture will consist of a series of recurrent convolutional units with ReLU alternated with max pooling followed by a series of recurrent up-convolution units with ReLU alternated with convolutional transformations with ReLU. There are residual connections between the contracting and expansive paths where the outputs of the recurrent convolutional units are concatenated with the outputs of the recurrent up-convolutional units, which helps with training deeper models. The model is called the R2U-Net because it combines a recurrent residual CNN with U-Net architecture.

**Results**

<img width="691" alt="Results Image" src="https://user-images.githubusercontent.com/39887209/211691884-226460dd-180c-4aee-abfb-85e6c27a86c2.png">
Our model achieved an average accuracy of around 0.72 with the losses descending and stabilizing towards the 60-90 range. Initially we had losses that were very high, but after identifying an error in our method of concatenation (we had been doing element-wise addition), we were able to reduce and stabilize the losses. We were also able to improve accuracy from 0.49 to 0.72.  


<img width="727" alt="Screen Shot 2023-01-10 at 6 46 53 PM" src="https://user-images.githubusercontent.com/39887209/211692077-7cf03917-ddab-464f-8f8f-09bcb02424cf.png">

The images above were outputted from our model following testing, with the model-produced segmented images on the left and the segmented labels on the right. The outlines of the segmentations looked pretty similar, and these qualitative results could be expanded upon by choosing more performance metrics beyond accuracy by which to evaluate the performance of the model. 

**Challenges**

There were a few notable challenges that we encountered. In preprocessing, we were able to diversify the data by creating flips, rotations, and cropping; however, we were unable to officially ‘augment’ the data by adding additional photos to the original dataset. This would require creating copies of images and labels in addition to the original dataset, causing potential increase in runtime. We also struggled to execute the concatenation portion of the U-Net architecture. This likely limited the accuracy since the U-Net is supposed to incorporate information from the down-sampling into the up-sampling blocks. Visualizing the results also posed a challenge because of the pixel representations being unusually low. This could probably be amended with more time, in which case we would attempt to multiply the values by a factor of 256 in order to create a stronger visual representation of the predicted pixels. 

**Reflection**

We consider the project to be a success. We achieved our base goal, which was to develop a model that can detect skin lesions from normal skin with high accuracy (greater than 0.5). However, re-implementing and debugging the model in Tensorflow proved to be significantly more time-consuming and difficult than initially anticipated. Thus, we did not have enough time to pursue our target and stretch goals, which were to create models that can detect whether a skin lesion is cancerous or not and determine the type of cancer, respectively.

Our approach was fairly dynamic, as we had to pivot around unanticipated challenges as we encountered them.  For example, in preprocessing we had to account for time efficiency because it initially took too long to run and eventually ran out of memory. Thus, we had to convert images into tensors using PIL and did rotations on the compressed versions of the images rather than the original images, which were quite large. We also had to reduce the number of channels in the model because of how long the training initially took  (8,16,32,64). Another modification we had to make was the elimination of concatenation in the architecture. Initially we ran into errors with the sizing, so we attempted to do a summation as a work-around. However, this ended up performing element wise addition that produced values that were inaccurate rather than expanding the width dimensionality of the upsampled blocks. Thus, we did not use concatenation in our implementation. 

In hindsight, running the original pytorch implementation on our local devices would’ve been a helpful guide for implementing the model ourselves. We had initially decided to skim the code and extract only the necessary parts; however, by referencing a working example we believe we would’ve developed a stronger intuition for the different classes/files. With more time we also would’ve looked to further diversify our training set with more rotations/reflections to prevent the model from overfitting to the data. We could also further tune the hyperparameters and parameters for each of the convolutional layers to produce a higher accuracy and better segmentation results. 

One takeaway we drew from this project was the importance of accounting for time-efficiency in dealing with image data. Similarly, we also modified the number of channels in the network (reducing to 8,16,32,64 rather than 64,128,256,512) in order to accommodate for more realistic time constraints. We learned that there is still a lot left to learn about the different model architectures that exist, their applications, and how they can be combined with other models to better achieve the desired goals (the paper we looked at compared the segmentation results across a variety of models and combinations of models.Implementing multiple techniques also helped us understand the importance of each step (i.e. what up-sampling, down-sampling, and concatenating intuitively represent) because of how we changed and modified the parameters when we ran our code. 

**Citations**

Alom, M. Z., Yakopcic, C., Hasan, M., Taha, T. M., & Asari, V. K. (2019). Recurrent residual U-net for medical image segmentation. Journal of Medical Imaging, 6(01), 1. https://doi.org/10.1117/1.jmi.6.1.014006 











