def train_generator(generator, discriminator_outputs, real_labels):
    """
    Arguments:
        generator: generator model object
        discriminator_outputs: ouput of the discriminator on a set of values z, D(G(z))
        real_labels: a vector of ones, size of discriminator_outputs

    Returns:
        g_loss: generator loss
    """
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss
