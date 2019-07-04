# Deep-learning-with-cats

This repository is a "toy" project so I can gain experience building deep neural networks. My first goal is generating pictures of cats using Generative Adversarial Networks. My second goal is making art with cats by applying styles to pictures of cats using deep convolutional neural networks. (^._.^)

**Update (2019/03/02): This contains a even more recent version of the code with even more features: https://github.com/AlexiaJM/relativistic-f-divergences **

Update (2018/11/02): See https://github.com/AlexiaJM/RelativisticGAN for a greatly enhanced version of the GAN codes, that incorporate all loss functions into a single file. It also includes additional better relativistic loss functions and many extra features (ex: Spectral normalization, Hinge Loss, Gradient penalty with any GAN loss, generate pictures every X iteration, learning rate decay, etc.). It still works by default to generate cats but it can also do CIFAR-10.

![](/images/DCGAN_220epochs.gif)

**Objectives**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * use **DCGAN** (Done)
  * use **WGAN** (Done)
  * use **WGAN-GP** (Done)
  * use **LSGAN** (Done)
  * use **BEGAN**
* Transform real cats into art pieces 
  * use **CycleGAN**
  * use **Fast neural style** (Done)
* Various/Others
  * Try adding Frechet Inception Distance (FID) as per https://arxiv.org/pdf/1706.08500.pdf
  * Try soft and noisy labels as per https://github.com/soumith/ganhacks
  * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
  
**Needed**

* Python 3.6, PyTorch, Tensorflow (for TensorBoard)
* Cat Dataset (https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html or http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd)
* TensorBoard logger (https://github.com/TeamHG-Memex/tensorboard_logger)

**To run**
```bash
$ # Download dataset and preprocess cat pictures 
$ # Create two folders, one for cats bigger than 64x64 and one for cats bigger than 128x128
$ sh setting_up_script.sh
$ # Move to your favorite place
$ mv cats_bigger_than_64x64 "your_input_folder_64x64"
$ mv cats_bigger_than_128x128 "your_input_folder_128x128"
$ # Generate 64x64 cats using DCGAN
$ python DCGAN.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder"
$ # Generate 128x128 cats using DCGAN
$ python DCGAN.py --input_folder="your_input_folder_128x128" --image_size 128 --G_h_size 64 --D_h_size 64 --SELU True
$ # Generate 64x64 cats using WGAN
$ python WGAN.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder"
$ # Generate 64x64 cats using WGAN-GP
$ python WGAN-GP.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder" --SELU True
$ # Generate 64x64 cats using LSGAN (Least Squares GAN)
$ python LSGAN.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder"
```

**To see TensorBoard plots of the losses**
```bash
$ tensorboard --logdir "your_input_folder"
```

# Results

**Discussion of the results at https://ajolicoeur.wordpress.com/cats.**

**DCGAN 64x64**

![](/images/DCGAN_209epoch.png)

**DCGAN 128x128 with SELU**

![](/images/DCGAN_SELU_128x128_epoch605.png)

**WGAN 64x64**

![](/images/WGAN_1408epoch.png)

**WGAN-GP 64x64 with SELU**

![](/images/WGAN_GP_iter15195.png)

**Fast style transfer**

![](/images/cat_style1.jpg)
![](/images/cat_style2.jpg)
![](/images/cat_style3.jpg)
![](/images/cat_style4.jpg)
![](/images/cat_style5.jpg)

![](/images/true_art.jpg)
