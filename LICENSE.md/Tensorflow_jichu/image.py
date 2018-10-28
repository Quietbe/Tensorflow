# coding=gbk
from PIL import Image
import numpy as np
import os.path


# import scipy

def loadImage():
    # 读取图片
    im = Image.open("images/0.jpg")
    im = im.resize((28,28),Image.BILINEAR)
    im.save('images/0_.jpg')
    # im = im.thumbnail((28,28))
    # im = Image.open(im).save('4_.png')
    im = im.convert("L")
    data = im.getdata()
    # data = np.matrix(data)
    #     print data
    data = np.reshape(data, (28, 28))
    # new_im = Image.fromarray(data)
    # 显示图片
    # new_im.show()
    print(data)
    # return data
loadImage()