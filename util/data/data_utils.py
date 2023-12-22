
def safe_crop_to_bounding_box(image, offset_h, offset_w, target_h, target_w):
    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]
    if offset_w + target_w > width:
        offset_w = width - target_w

    if offset_h + target_h > height:
        offset_h = height - target_h

    return image[offset_h:offset_h + target_h, offset_w:offset_w + target_w]
