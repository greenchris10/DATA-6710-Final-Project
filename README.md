# DATA-6710-Final-Project

The paper I chose for my project:

"Semantic Image Completion and Enhancement using GANs" (https://arxiv.org/pdf/2307.14748.pdf)

The paper focuses on image completion with WGANs.

<img width="1021" alt="Screenshot 2023-08-17 at 12 04 47 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/2e46e89c-e64b-4890-a250-09cb362c549e">

Generative Adversarial Network (GAN) is a type of machine learning model that consists of two parts: a generator and a discriminator. GANs are used for generating new data that is similar to a given set of training data.

The key idea of a GAN is that these two parts are in a constant competition or "adversarial" relationship:

The generator tries to create data that is so realistic that the discriminator can't tell it apart from the real data.
The discriminator tries to become better at telling real data from generated data.
During training, they go back and forth in a loop:

The generator creates fake data.
The discriminator evaluates the real and fake data, providing feedback to the generator about how to improve.
The generator adjusts its parameters to make the generated data more convincing.
The discriminator improves its ability to distinguish between real and fake data.

Wasserstein GANs (WGANs)

<img width="588" alt="Screenshot 2023-08-17 at 12 23 52 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/30f5540b-82d9-498a-a52c-85e55e4033e3">

From a high level, Wasserstein GANs (WGANS) are essentially the same as normal GANs, but they use Wasserstein distance as a loss function. This loss function measures the distance between real and generated data in terms of how much work is required to transform one set of data into the other. To put it simply, WGANs aim to minimize the distance between real and generated images rather than minimizing the distance between the probabilities of real and generated images.

The paper uses a WGAN for image completion. Here are some expamples of their results:

<img width="461" alt="Screenshot 2023-08-17 at 12 30 23 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/07bf0723-5167-4ef8-9660-f5c7f8ae058f">

<img width="668" alt="Screenshot 2023-08-17 at 12 32 22 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/fd25e320-6ded-4327-a2c6-d768083ea254">

<img width="583" alt="Screenshot 2023-08-17 at 12 32 54 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/272c0108-fdf7-416c-a77b-b1889b1a7483">

My implementation process:

The first step I had to do was find a dataset. I used the celeA dataset and applied a binary mask to all the images:

<img width="752" alt="Screenshot 2023-08-17 at 1 46 06 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/5218e54c-b0ac-44df-9e13-610557fefe9f">

<img width="776" alt="Screenshot 2023-08-17 at 1 35 04 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/498b1fd1-38d2-4069-8caa-b9700345eb82">

GAN Architecture from a high level:

This wasn't exactly my architecture but similar idea-

<img width="1168" alt="Screenshot 2023-08-17 at 1 49 09 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/a3a2a5a3-929d-4222-a940-78af098aa451">

I was referencing code from the lecture and tensorflow's website.
(https://www.tensorflow.org/tutorials/generative/dcgan)

Implementing the generator:

<img width="693" alt="Screenshot 2023-08-17 at 12 50 32 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/b24101a1-191a-42a9-a85a-56f4d5613b64">
<img width="702" alt="Screenshot 2023-08-17 at 12 51 00 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/3faae117-a1ee-48fc-a2a8-02cde4dadc8a">

Batch Normaliztion is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.

Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. Deep neural networks utilizing Leaky ReLU were found to reach convergence slightly faster than those using ReLU.

Implementing the discriminator:

<img width="657" alt="Screenshot 2023-08-17 at 12 54 50 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/87aa4541-512e-4f14-88c7-2e5b8a51c674">

Implementing the loss functions:

<img width="1131" alt="Screenshot 2023-08-17 at 1 09 08 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/377e4659-fdca-4027-83f0-7b357712f388">

Problems I encountered:

-Wasserstein GANs (any image to image GANs) are particularly difficult to implement.

-One of the biggest challenges for these networks is establishing a proper balance between the two opposing forces. The increase in performance for the discriminator negatively affects the performance of the generator and vice versa, so finding a state of equilibrium is difficult. Much of the effort put into constructing GANs revolves around determining when the optimal amount of training has been reached.

-My code would run but wouldn't properly train the model. I also learned that I was trying to implement a DCGAN which may not have been applicable to the type of problem I was trying to solve.

-I wasn't able to correctly implement the random noise that is typically added to the latent space vectors for the generator.

-Image to Image GANs are really strict with the type of data that is used to train the model. A lot of the image to image GANs I found had specific data sets that were inteded to be used with them.

Working with Pix2pix:

Pix2pix is a conditional generative adversarial network (cGAN) created in 2017. It can perform image to image tasks such as synthesizing photos from label maps, generating colorized photos from black and white images, turning Google Maps photos into aerial images, and even transforming sketches into photos.

Image to image GANs typically use a UNet architecture:

<img width="642" alt="Screenshot 2023-08-17 at 2 44 52 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/822fae4e-a2f0-4861-984d-86427355ad58">

I was suggested by the professor to use the pix2pix model from tensorflow on my masked image dataset.

I tried but the result was just a (64,64) static image.

I used two of the datasets they provided and got the following results:

<img width="583" alt="Screenshot 2023-08-17 at 2 35 00 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/7f1fd2e9-b7cb-49d5-950a-ddc708ec756f">

<img width="609" alt="Screenshot 2023-08-17 at 2 35 25 PM" src="https://github.com/greenchris10/DATA-6710-Final-Project/assets/120329214/c3441f6b-12fb-4ea0-addb-d5dbda5b651b">

I also managed to find a sketch to portrait dataset but it was 20gbs and it wasn't one of the 6 that tensorflow provided.
