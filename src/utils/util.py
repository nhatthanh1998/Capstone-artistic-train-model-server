def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def training_transform():
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform


def style_transform():
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform
