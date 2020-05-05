from tensorflow.keras import backend as K

def content_loss(img, content):
    # Sum up the pixel-wise squared difference between img and the content image
    return K.sum(K.square(img - content))

def get_gram_matrix(img):
    gram = K.dot(img, K.transpose(img))
    return gram

def style_loss(img, style, img_h, img_w, num_channels=3):
    gram_style = get_gram_matrix(style)
    gram_img = get_gram_matrix(img)
    size = img_h * img_w
    loss = K.sum(K.square(gram_style - gram_img)) / (4.0 * (num_channels**2) * (size**2))
    return loss