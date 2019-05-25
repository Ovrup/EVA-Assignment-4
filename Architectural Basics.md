# Architectural Basics

### 5. Receptive field:

The 1st thing before creating any architecture to think is the receptive field. In order to detect any object the network should see the whole image & the receptive field at final layer should match the input size.

Say we have an input image of size 7 * 7. If we perform convolution with 3 * 3 kernel 3 times, the output will be 1 * 1 & the receptive field in final layer will be 7 * 7. It means the network now sees the whole image & can predict what the image is about.



## 1. How many layers:

Before choosing the number of layers in a network we should think couple of things.

First, the objective is to detect an object in image. In the initial layers the kernels detect edges & gradients. In the middle layers the kernels combine the features i.e edges & gradients from previous layer & build textures. Then next layers would detect patterns , parts of objects & finally the whole object. So, the network should deep enough to detect these features. Also the receptive field criteria should match.



## 8. Kernels & how to decide the number of kernels :

In the initial layers the network detects edges & gradients. For that purpose 32 is a good number to start & then we gradually increase the number layer by layer to 64 to 128 and so on. But for simple data sets like MNIST we can start with 16 or even 10 for first layers.



## 4. 3 * 3 Convolution:

3 * 3 convolution is performed by the kernel of size 3 * 3 on the input to extract features. Normally the 3 * 3 convolution is used in convolution block followed by transition blocks.



## 3. 1 * 1 convolution :

This is a special type of convolution which helps to reduce the number of parameters. Say we have 3 convolution layers with 64, 128 & 256 no of kernels respectively in convolution block. If we keep moving like this eventually  we will end up with large number of parameters. Here we use 1 * 1 convolution to reduce the no of channels to lower values, say 64. 1 * 1 convolution is added in transition layers.



## 2. Max Pooling:

Max pooling is used when we need to down sample the image size. Down sampling is important because it helps to reduce the no of layers in the network. For example , we have an input image of size 400 * 400 & we are using 3 * 3 convolution. In order to reach the final layer(receptive field of 400 * 400) we have to add 200 layers. But max pooling helps to reduce this to a very small number(for example 20 layers)



## 11. Position of Max Pooling:

The position of max pooling can be in the transition block. The transition block is meant to reduce the image dimension. 1 * 1 convolution helps to reduce the image depth i.e no of channels. Max pooling is used to reduce the image width & height. After every convolution block there can be this transition block which uses max pooling. Also the max pooling should not be used in few layers before the output layer for making sure all information in those last layers are conveyed to the output.



## 17. The distance of Max Pooling from Prediction:

 Max pooling should not be used in 1 or 2 layers before the output/prediction layer for making sure all information in those last layers are conveyed to the output.



## 12. Concept of transition layer:

The transition layer reduces the image dimensions which helps in reducing the no of layers as well as parameters. In transition layers 1 * 1 convolution is performed which is followed by max pooling. 1 * 1 convolution reduces the no of image channels. Max pooling reduces the image height & width.



## 13. Position of transition layers:

The transition layers are placed after each convolution blocks in the network. It makes sure that the output of previous convolution blocks are down sampled & then fed to the next convolution block with fixed no of kernels as the number of kernels are always fixed for convolution block.



## 6. Softmax:

Softmax is the activation function which gives the final prediction in the output layer. The output of softmax is in between 0-1. 



## 19. When do we stop convolutions and go ahead with a larger kernel or some other alternative:

When we reach at a layer where the image resolution is around 7 * 7, we should stop convolution & use large kernel to reach output. The reason is at this small resolution there is no feature could be seen by the kernel & we don’t want the image to be more quantized by convolution. It would unnecessarily create more layers & increase no of parameters.



## 21. Batch size & effect of batch size:

The batch size should always be more then the no of classes present in a data set. This is to make sure that in one batch all the classes are covered & weights are updated accordingly. This will generalise the model highly.



## 14. Number of epochs & when to increase them:

The network should be trained for sufficient number of epochs so that it can optimize the weights properly to reach the minimum point of cost function. If we set epoch small then the model would not be trained properly & for large no of epoch the model may overfit. We should observe the model behaviour with various epoch values & chose a number accordingly. Also we can use regularisation and train the model as many no of epochs.



## 22. When to add validation checks:

Validation check is added to monitor the models performance while training. It helps us to detect whether our model is overfitting. Given the hyperparameter values, we use training set to train our model. But we don’t know which values of hyperparameters are best suit for the model. Here validation data comes in picture. Validation data is used for evaluating the model performance & based on model’s performance on validation set we can try different values for hyperparameters & keep the best trained model.



## 20. How do we know our network is not going well, comparatively, very early:

The easiest way to evaluate our model’s performance compared to other model is to check the validation accuracy for first 2 or 3 epochs. If the validation accuracy of our model is not good enough compared to the other it means it will never compete with it. We should try to tweak the model hyperparameters to improve the validation accuracy.



## 7. Learning Rate:

This is one of the hyperparameters of neural networks which decides the step size in gradient descent algorithm. It controls how fast or slow the training would be. 



## 23. LR schedule & concept behind it:

To decide a value of learning rate for training a model is sometimes very difficult. Sometimes we select a constant value of  learning rate & don’t know whether it is suitable for our model or not. As we move closer to the minimum of cost function our step size should get smaller. If we end up with large LR then we could miss the minimum & overshoots. To avoid this we use LR scheduler which helps us reducing the LR as we move towards the minimum point. LR schedule is a callback function which reduces the LR after every epoch at a particular decay rate defined.



## 9. Batch Normalization:

Batch normalization technique help to normalise the batches of images by adjusting & scaling them.  For example, for the images some features may have low pixel intensity (1-10) & some may have high pixel intensity. So, batch normalization scales them & makes it smooth. It helps training process. It also reduces overfitting. Batch normalization layer is added after every convolution layer.



## 18. The distance of batch normalization from prediction:

Batch normalization layer should not be added just before the output/ prediction layer. It makes sure that all information from penultimate layer are conveyed to output.



## 15. Dropout:

Dropout is a regularization technique which helps to reduce the gap between training accuracy & validation accuracy. Dropout randomly drops some of the kernels from the layer & train the model with rest of the kernels. But for validation all the kernels work which reduces the gap. Dropout normally added after (convolution –> Batch normalization).



## 16. When do we introduce dropout :

We introduce dropout if we see the model overfits. The model overfits when it learns all details & noise in training data. These features in training data may not be relevant for new data in test set but negatively impacts the validation accuracy. 

In such case dropout randomly removes some of the kernels while training makes sure the model generalise the training data.



## 10. Image normalization:

Unlike batch normalization, Image normalization applies to single images. It takes minimum & maximum pixel values from an image, consider the minimum pixel values as 0, maximum pixel value as 255 & stretches the whole image between 0 & 255. This is not frequently used in deep learning models.



## 24. Adam vs SGD:

The main difference between Adam & SGD is that SGD has a slower convergence rate than Adam. Also adam has lowest training error but not validation error.  