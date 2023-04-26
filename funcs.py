import numpy as np


def draw_mulit_images_in_one(images: list[np.ndarray], width_repeat: int = 5, line_width: int = 3) -> np.ndarray:
    """将多张图片绘制到一张图片

    Args:
        images (list[np.ndarray]):    绘制的图片列表,所有图片大小相同
        width_repeat (int, optional): x轴重复次数. Defaults to 5.
        line_width (int, optional):   图片之间的线宽. Defaults to 3.

    Returns:
        np.ndarray: 绘制的图片
    """
    import math
    height_repeat = math.ceil(len(images) / width_repeat)

    height, width, channel = images[0].shape
    # new image
    new_height = height * height_repeat + line_width * (height_repeat - 1)
    new_width  = width  * width_repeat  + line_width * (width_repeat  - 1)
    palette = np.zeros((new_height, new_width, channel), dtype=np.uint8)

    for i, image in enumerate(images):
        height_index = math.floor(i / width_repeat) # h index
        width_index  = i % width_repeat             # w index

        # x y 坐标
        height_1 = height_index * (height + line_width)
        height_2 = height_1 + height
        width_1  = width_index  * (width + line_width)
        width_2  = width_1 + width

        palette[height_1:height_2, width_1:width_2] = image

    return palette