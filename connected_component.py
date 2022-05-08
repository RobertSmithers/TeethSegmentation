import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.color import label2rgb
from skimage import measure

"""###Post-process images###"""

#connected components and watershed
def ConnectedComp(image):
  # Load in image, convert to gray scale, and Otsu's threshold
  kernel =(np.ones((3,3), dtype=np.float32))
  image = cv2.imread(img)
  image=cv2.resize(image,(3130,1200))
  image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  erosion = cv2.erode(thresh,kernel,iterations =5)
  #gradient, aka the contours
  gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)

  ret,markers=cv2.connectedComponents(erosion)
  new=watershed(erosion,markers,mask=thresh)
  RGB = label2rgb(new, bg_label=0)
  return erosion,gradient,RGB

#try a random image to test
img='/content/98.png'
erosion,gradient,RGB=ConnectedComp(img)

fig = plt.figure(figsize = (16,16))
ax = fig.add_subplot(1, 3, 1) 
plt.imshow(erosion)
ax = fig.add_subplot(1,3, 2) 
plt.imshow(gradient)
ax = fig.add_subplot(1,3, 3) 
plt.imshow(RGB)
plt.show()
