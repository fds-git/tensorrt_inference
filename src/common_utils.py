def image_shape_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    image_shape =  tuple(mapped_int)
    assert len(image_shape) == 4
    return image_shape