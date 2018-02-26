# NMT_GAN
This is an open source implementation of our framework to improve NMT with conditional sequence generative adversarial nets.which is described in the following papers: 

Yang, Z., Chen, W., Wang, F., & Xu, B. Improving neural machine translation with conditional sequence generative adversarial nets. (NAACL 2018)

Requirements: Tensorflow 1.2.0, python 2.x

Useage:

pre-train the discriminator by: sh discriminator_pretrain.sh

pre-train the generator by: sh train.sh

generate the samples by: sh generate_sample.sh

run the gan training by: sh gan_train.sh

