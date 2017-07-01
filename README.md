# Deep-learning-with-cats

This repository is a "toy" project so I can gain experience building deep neural networks. I have a few subprojects in mind. My first goal is learning to generate pictures of cats with Generative Adversarial Networks (^._.^). 

![](/images/DCGAN_220epochs.gif)

**Objectives (so far)**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * use DCGAN (Done)
  * use WGAN (Done)
  * use WGAN-IP (In progress)
  * use BEGAN
* Various/Others
  * Preprocess cat images so we get aligned cat faces for much better GAN convergence (Done)
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
  * Try making higher resolutions pictures (Limited by 6gb of GPU RAM)
  * Try Self-Normalizing Neural Networks (SELU) as per https://arxiv.org/abs/1706.02515
  * Try soft and noisy labels as per https://github.com/soumith/ganhacks
  * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
  
**Needed**

* Python 3.6, PyTorch, Tensorflow (for TensorBoard)
* Cat Dataset (https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html)
* TensorBoard logger (https://github.com/TeamHG-Memex/tensorboard_logger)

**To run**
```bash
$ # Download dataset and preprocess cat pictures (folder "cat_dataset_output" contains the cat faces)
$ sh setting_up_script.sh
$ # Generate cats using DCGAN
$ python Meow_DCGAN.py --input_folder "your_input_folder" --output_folder "your_output_folder"
$ # Generate cats using WGAN
$ python Meow_WGAN.py --input_folder "your_input_folder" --output_folder "your_output_folder"
```
**To see TensorBoard plots of the losses**
```bash
$ tensorboard --logdir "your_input_folder"
```

# Results

**DCGAN**

It converges to very realistic pictures in about 2-3 hours with only 209 epochs but some mild tweaking is necessary for proper convergence. The sweet spot is using .00005 for the Discriminator learning rate and .0002 for the Generator learning rate; this prevents the Discriminator from getting too good. There's no apparent mode collapse and we end up with really cute pictures!

![](/images/DCGAN_209epoch.png)

**WGAN**

It converges but very slowly (took 4-5h, 600+ epochs) and only when using 64 hidden nodes. I could not make the generator converge with 128 hidden nodes. There is some pretty striking mode collapse here, many cats have heterochromia, one eye closed and one eye open or a weird nose. Overall the results are not as impressive as with DCGAN but then it could be because the neural networks are less complex so this might not be a fair comparison. This needs to be attempted again with WGAN-improved because these convergence issues could likely be due to the weight clipping. In the paper on improving WGAN by Gulrajani et al. (2017), they were able to train a 101 layers neural network to produce pictures! So I doubt that training a cat generator with 5 layers and 128 hidden nodes would be much of a problem. So far though, WGAN is disapointing.

![](/images/WGAN_1408epoch.png)
