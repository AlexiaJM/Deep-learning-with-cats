# Deep-learning-with-cats

This repository is a "toy" project so I can gain experience building deep neural networks. I have a few subprojects in mind. My first goal is learning to generate pictures of cats with Generative Adversarial Networks (^._.^). 

![](/images/DCGAN_220epochs.gif)
![](/images/DCGAN_209epoch.png)

**Objectives (so far)**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * Preprocess cat images so we get aligned cat faces for much better GAN convergence (Done)
  * use DCGAN (Done)
    * Basic implementation (Done)
    * Keeping log for TensorBoard (Done)
    * Automatic folder setup (Done)
    * Tune hyperparameters (Done)
    * Tweak model structure, maybe LeakyReLU and dropouts in G
  * use WGAN
  * When it will be possible to implement in Pytorch (not currently possible), use WGAN-IP
  * use BEGAN
* Various/Others
  * Try soft and noisy labels as per https://github.com/soumith/ganhacks
  * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
  * Try different learning rates for G and D (Done, so far 1/4-1/5 of G for D works best)
  * Try skipping Discriminator learning every 2 iteration as per https://github.com/carpedm20/DCGAN-tensorflow
  * Add multi-gpu and non-CUDA option (Done)
  * Option to load previous models (Done)
  * Add log to output (Done)
  * Identify and remove outliers
    * Remove obvious outliers manually (Done)
    * Find outliers based on a certain measure
  
**Needed**

* CUDA GPU
* CUDA, CDNN, Python 3.6, PyTorch, Tensorflow (for TensorBoard)
* Cat Dataset (https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html)
* TensorBoard logger (https://github.com/TeamHG-Memex/tensorboard_logger)

**To run**
```bash
$ # Download dataset and preprocess cat pictures (folder "cat_dataset_output" contains the cat faces)
$ sh setting_up_script.sh
$ # train GAN network
$ python Meow_DCGAN.py --input_folder "your_input_folder" --output_folder "your_output_folder"
```
**To see TensorBoard plots of the losses**
```bash
$ tensorboard --logdir "your_input_folder"
```
