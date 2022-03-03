import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

pic = Image.open('Dark.jpg')
pic_arr = np.asarray(pic)


LogTransformation = lambda i: (255/float(math.log10(255*0.08+1)))*float(math.log10(0.08*i+1))
Loged = np.vectorize(LogTransformation)
result = Loged(pic_arr).astype(int)
plt.imshow(result)
plt.show()

