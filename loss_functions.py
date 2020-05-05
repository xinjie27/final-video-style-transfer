from tensorflow.keras import backend as K

IMG_HEIGHT = 0
IMG_WIDTH = 0

def content_loss(img, content):
    # Sum up the pixel-wise squared difference between img and the content image
    return K.sum(K.square(img - content))

def get_gram_matrix(img):
    # TODO: May require debugging
    gram = K.dot(img, K.transpose(img))
    return gram

def style_loss(img, style, num_channels=3):
    gram_style = get_gram_matrix(style)
    gram_img = get_gram_matrix(img)
    size = IMG_HEIGHT * IMG_WIDTH
    loss = K.sum(K.square(gram_style - gram_img)) / (4.0 * (num_channels**2) * (size**2))
    return loss

def total_loss(img, content, style, alpha=10, beta=40):
    l_content = content_loss(img, content)
    l_style = style_loss(img, style)
    loss = alpha * l_content + beta * l_style
    return loss