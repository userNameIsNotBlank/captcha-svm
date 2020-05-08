import time

import numpy

from PIL import Image
import matplotlib.pyplot as plt
import img_tools
import svm_features


def get_bin_table(threshold=140):
    """
    获取灰度转二值的映射table
    :param threshold:
    :return:
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return table


image = Image.open("E:/code/python/python3/machine_learning/captcha-svm/imgs/6937.png") # type: Image.Image

imgry = image.convert('L')


table = get_bin_table()

out = imgry.point(table, '1')



# for j in range(0, out.height):
#     wStr = ''
#     for i in range(0, out.width):
#         if out.getpixel((i, j))>=0 and out.getpixel((i, j))<=255:
#             wStr += str(out.getpixel((i, j)))
#     print(wStr)

out = img_tools.get_clear_bin_image(out)

feature = svm_features.get_feature(out)
print(feature)

def demo():
    print(">>> Start")
    for i in range(789):
        if i % 10 == 0:
            time.sleep(0.2)
            print("\r进度： {:.2f}%".format(100 * i / 788), end='')
    print("\r进度： 100.00%")
    print('>>> Done!')


if __name__ == '__main__':
    demo()