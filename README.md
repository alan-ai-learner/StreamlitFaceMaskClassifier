# Streamlit Face Mask Classifier
## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Dependencies](#dependencies)
  * [Credit](#credit)
  
 # Demo
 - Take a look at the project.[View](https://face-mask-classifier.herokuapp.com/)
 
# Overview
- This face mask classifier is made by using **VGG-19** Convolution Neural Network using pretrained imagenet weights.
- Having an accuracy of 95% and loss is 0.45.
- This model is trained on 20k images have 10k images in each class.

# Motivation
- In this lockdown and covid-19 period i think i can use my **Computer Vision Skills** to made a classifier which can be used to classify the images tht are earing mask or not.

# Dependencies

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) [<img target="_blank" src="https://miro.medium.com/max/984/1*78KMHVVecnTzkLbN7xW0tQ.png" width=200>](https://www.tensorflow.org/guide/effective_tf2) [<img target="_blank" src="https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e18182ad27bcfbb9dff263a_RGB_Logo_Horizontal_Color_Light_Bg.png" width=200>](https://www.streamlit.io/) [<img target="_blank" src="data:image/png;base64,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" width=200>](https://www.heroku.com/)

## Team
 [<img target="_blank" src="https://user-images.githubusercontent.com/53686128/85886789-51348d80-b804-11ea-8a8f-23567029cfe1.jpeg" width=200>]()


## Credit
 Google Images and my scrapping images python scripts.
 
 
