# Pix2Pix CarMask

This was inspired by a [Kaggle Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/kernels). In this competition, you’re challenged to develop an algorithm that automatically removes the photo studio background. This will allow Carvana to superimpose cars on a variety of backgrounds. You’ll be analyzing a dataset of photos, covering different vehicles with a wide variety of year, make, and model combinations.

Although there are many ways to get the job done. I though I would try [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) to do it. 

## Training

The dataset contains 5088 photos of cars and a 5088 paired masked image of the photos

![](/images/Train_Loss.png)


## Test result

Predicted masks given a photo of a car

![](/images/Input_Image.gif) 

![](/images/Gen_Image.gif)


## Masked Images

After masking was applied using the input images and the predicted masks using Pix2Pix

![](Masked_87.png)


![](Masked_88.png)


![](Masked_89.png)


![](Masked_90.png)


![](Masked_91.png)


