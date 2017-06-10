# Deep-learning-with-cats

I am trying to learn deep learning on a practical level and this repository is for that purpose. I have a few subprojects in mind, all having to do with cats. My first goal is learning to generate pictures of cats with Generative Adversarial Networks (^._.^). 

**Using**
* Python 3.6
* Pytorch
* Geforce 1060

**Dataset**

*Cat Dataset*

You can download it here:

https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/catData.html

**Objectives (so far)**
* Generate images of cats using various types of Generative Adversarial Networks (GAN)
  * with DCGAN (In Progress)
    * Basic implementation (Done)
    * Keeping log for TensorBoard (Done)
    * Automatic folder setup (Done)
    * Tune hyperparameters (In progress)
    * Tweak model structure (maybe LeakyReLU and dropouts in G)
    * Try soft and noisy labels as per https://github.com/soumith/ganhacks
    * Try adding decaying noise to input as per https://github.com/soumith/ganhacks
    * Try different learning rates for G and D
  * with WGAN
  * When it will be possible to implement in Pytorch (not currently possible), try WGAN-IP
  * with BEGAN
