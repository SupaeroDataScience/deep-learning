def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):
    """
    Arguments:
        discriminator: discriminator model object
        images: a batch of data from the dataset
        real_labels: a vector of ones, size of images
        fake_images: a batch of images generated by the generator
        fake_labels: a vector of zeros, size of fake_images

    Returns:
        d_loss: discriminator loss
        real_output: output of the discriminator on the real images
        fake_output: output of the discrimiator on the fake images
    """
    discriminator.zero_grad()
    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels)
    real_output = outputs

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels)
    fake_output = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_output, fake_output
