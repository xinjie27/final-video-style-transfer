from tensorflow.keras import backend as K

IMG_HEIGHT = 0
IMG_WIDTH = 0

def content_loss(img, content):
    # Sum up the pixel-wise squared difference between img and the content image
    return K.sum(K.square(img - content))

def get_gram_matrix(img):
    """
    Compute the gram matrix G for an image tensor
    """
    if K.image_data_format() == "channels_first":
        features = K.batch_flatten(img)
    else:
        features = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    G = K.dot(features, K.transpose(features))
    return G

def style_loss(img, style, num_channels=3):
    gram_style = get_gram_matrix(style)
    gram_img = get_gram_matrix(img)
    size = IMG_HEIGHT * IMG_WIDTH
    loss = K.sum(K.square(gram_style - gram_img)) / (4.0 * (num_channels**2) * (size**2))
    return loss

def total_loss(img, content, style, alpha=10, beta=80):
    l_content = content_loss(img, content)
    l_style = style_loss(img, style)
    loss = alpha * l_content + beta * l_style
    return loss