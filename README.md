# CrossViT Sample code

CrossViT is a Cross-Attention Multi-Scale Vision Transformer for image classification. This combined the image with different patches size to extract the multiscale feature representation.

To leverage the properties to extract strong visualization feature, This codes implements CrossViT model for generative tasks, where the input feature dimension and the output dimension have same shape. My leveraging its properties of understanding the global context, could help maintain visually consistent output.

### Parameter Output:

---

Trainable Parameters: 99.107M

Shape of out : torch.Size([1, 3, 256, 256])

---

### Possible Improvements:

---

1. CrossViT Generator is great for global context of the image but it cannot capture local features.
2. We can use a combination of CrossViT and Convolutional Neural Network to capture both global and local features.
3. We can try different type of attention mechanism (only the example: Axial Attention) to capture the global context as well as local features of the image.
4. For research purposes, I have used CrossViT as a generator as well as a discriminator in the GAN model. Finally, I have used the Attention U-Net based as a generator to capture local features.
5. CloVe model (Discussed during the interview) can be used to capture the local features of the image for image generation tasks.
