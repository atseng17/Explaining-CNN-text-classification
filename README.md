# Explaining-CNN-text-classification


In this repo we show how a CNN model, with a given input can be explained. The CNN model used in this repo is a pytorch implementation of [1]. The model is first trained on the polarity dataset and then explained via an adverserial attack [2] on the word embeddings.

<img src="demo_images/Fig1.png" height="150">

### How to run
``` 
# install required packages
conda env create -f exp_text.yml

# run demo on jupyter notebook
jupyter notebook
```




## References
<a id="1">[1]</a> 
Kim, Yoon (2014). 
Convolutional Neural Networks for Sentence Classification.
EMNLP, 11(3), 1746-1751.

<a id="2">[2]</a> 
Arash Rahnama, Andrew Tseng (2021).
An Adversarial Approach for Explaining the Predictions of Deep Neural Networks
CVPR Workshops, 2021, pp. 3253-3262.