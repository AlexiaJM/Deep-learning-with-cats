# Deep-learning-with-cats

I am trying to learn deep learning on a practical level and this repository is for that purpose. I have a few subprojects in mind, all having to do with cats. My first goal is learning to generate pictures of cats with Generative Adversarial Networks (^._.^). 

**Objectives (so far)**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * Preprocess cat images as per original paper https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/ECCV_CAT_PROC.pdf so we get aligned cat faces for much better GAN convergence (*Priority to be done)
  * use DCGAN (In Progress)
    * Basic implementation (Done)
    * Keeping log for TensorBoard (Done)
    * Automatic folder setup (Done)
    * Tune hyperparameters and get it to converge! (In progress)
    * Tweak model structure (maybe LeakyReLU and dropouts in G)
    * Try soft and noisy labels as per https://github.com/soumith/ganhacks
    * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
    * Try different learning rates for G and D
    * Try skipping Discriminator learning every 2 iteration as per https://github.com/carpedm20/DCGAN-tensorflow
  * use WGAN
  * When it will be possible to implement in Pytorch (not currently possible), use WGAN-IP
  * use BEGAN

**Needed**

* CUDA GPU
* CUDA, CDNN, Python 3.6, PyTorch, Tensorflow (for TensorBoard)
* Cat Dataset (https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html)
* TensorBoard logger (https://github.com/TeamHG-Memex/tensorboard_logger)

**To run**
```bash
$ python Meow_DCGAN.py --input_folder "your_input_folder" --output_folder "your_output_folder"
```
