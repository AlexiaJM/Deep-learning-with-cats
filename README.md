# Deep-learning-with-cats

This repository is a "toy" project so I can gain experience building deep neural networks. My first goal is learning to generate pictures of cats with Generative Adversarial Networks (^._.^). 

![](/images/DCGAN_220epochs.gif)

**Objectives (so far)**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * use **DCGAN** (Done)
  * use **WGAN** (Done)
  * use **WGAN-IP** (In progress)
* Various/Others
  * Preprocess cat images so we get aligned cat faces for much better GAN convergence (Done)
  * Separate cats by size to be better able to generate cats of certain sizes (Done)
  * Fix DCGAN models so that they can adapt to different input image sizes (Done)
  * Keeping log for TensorBoard (Done)
  * Automatic folder setup (Done)
  * Add multi-gpu and non-CUDA option (Done)
  * Option to load previous models (Done)
  * Add log to output (Done)
  * Identify and remove outliers
    * Remove obvious outliers manually (Done)
    * Find outliers based on a certain measure
  * Tweak models structures, maybe LeakyReLU and dropouts in G
  * **Try making higher resolutions pictures** (Limited by 6gb of GPU RAM)
    * 128 x 128 (Done)
    * 256 x 256 
  * **Try Self-Normalizing Neural Networks (SELU)** as per https://arxiv.org/abs/1706.02515 (Done)
  * Try adding Frechet Inception Distance (FID) as per https://arxiv.org/pdf/1706.08500.pdf
  * Try soft and noisy labels as per https://github.com/soumith/ganhacks
  * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
  
**Needed**

* Python 3.6, PyTorch, Tensorflow (for TensorBoard)
* Cat Dataset (https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html)
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
$ python Meow_DCGAN.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder"
$ # Generate 128x128 cats using DCGAN
$ python Meow_DCGAN.py --input_folder="your_input_folder_128x128" --image_size 128 --G_h_size 64 --D_h_size 64 --SELU True
$ # Generate 64x64 cats using WGAN
$ python Meow_WGAN.py --input_folder "your_input_folder_64x64" --output_folder "your_output_folder"
```
**To see TensorBoard plots of the losses**
```bash
$ tensorboard --logdir "your_input_folder"
```

# Results

**DCGAN**

It converges to very realistic pictures in about 2-3 hours with only 209 epochs but some mild tweaking is necessary for proper convergence. You must choose separate learning rates for D and G so that neither G or D become way better than the other, it's a very careful balance but once you got it, you're set for convergence! With 64 x 64 images, the sweet spot is using .00005 for the Discriminator learning rate and .0002 for the Generator learning rate. There's no apparent mode collapse and we end up with really cute pictures!

![](/images/DCGAN_209epoch.png)

**High Resolution DCGAN and SELU**

All my initial attempts at generating cats in 128 x 128 with DCGAN failed. However, simply by replacing the batch normalizations and ReLUs with SELUs, I was able to get slow (6+ hours) but steady convergence with the same learning rates as before. SELUs are self-normalizing (see Klambauer et al.(2017)) and thus remove the need of batch normalization. It is very fascinating as SELUs are extremely new (one month old) so no research has been done on SELUs and GANs but from what I observed, they seem to greatly increase GANs stability.

![](/images/DCGAN_SELU_128x128_epoch605.png)

**WGAN**

It converges but very slowly (took 4-5h, 600+ epochs) and only when using 64 hidden nodes. I could not make the generator converge with 128 hidden nodes. With DCGAN, you have to tweak the learning rates a lot but you are able to see quickly if it's not going to converge (If Loss of D goes to 0 or if loss of G goes to 0 at the start) but with WGAN, you need to let it run for many epochs before you can tell. 

Visually, there is some pretty striking mode collapse here; many cats have heterochromia, one eye closed and one eye open or a weird nose. Overall the results are not as impressive as with DCGAN but then it could be because the neural networks are less complex so this might not be a fair comparison.

This needs to be attempted again with WGAN-GP because these convergence issues could likely be due to the weight clipping. In the paper on improving WGAN by Gulrajani et al. (2017), they were able to train a 101 layers neural network to produce pictures! So I doubt that training a cat generator with 5 layers and 128 hidden nodes would be much of a problem. So far though, WGAN is disapointing.

From reading the paper "GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium" by Heusel et al. (2017), it seems that the Adam optimiser has some properties which make mode collapse much less likely and it lower the chance of getting stuck into a bad local optimum. This is likely contributing to the problem with WGAN which doesn't use Adam; considering that WGAN-GP use Adam, it is much more likely to give good results like DCGAN.

![](/images/WGAN_1408epoch.png)
