#importing numpy module
import numpy as np 

import matplotlib.pyplot as plt
#Matplotlib is a plotting library for the Python

#importing cv
import cv2
#Open Source Computer Vision Library 
#used for computer vision and image processing.

print("loading models.....")

#taking image name as input from the user
print("Enter the image name you want to colorize:")
image = input()

#the name with which the user wants to store the colorized version
print("Enter the image name with which you want to save the colorized image:")
name2 = input()

#loading our Caffe model.
prototxt = 'colorization_deploy_v2.prototxt'
caffe_model = 'colorization_release_v2.caffemodel'
net = cv2.dnn.readNetFromCaffe( prototxt, caffe_model)
#The function cv2.dnn.readNetFromCaffe() accepts two parameters:
#prototxt – path to “.prototxt” file
#caffe_model – path to “.caffemodel” file

pts = np.load('pts_in_hull.npy')
#loaded the “.npy” file using NumPy.

#Pass the image through the layers
#get the layer id from the caffee model by using the function “.getLayerId()”. The “.getLayerId()” takes one parameter.Eg net.getLayerId(“name of the layer”)
class8 = net.getLayerId("class8_ab")#layer1
print(class8)
#Get the serial number id of the layer according to the name of the layer.
conv8 = net.getLayerId("conv8_313_rh")#layer2
print(conv8)
pts = pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]


#reading image
image = cv2.imread(image)
plt.imshow(image)
plt.title('image')
plt.show()

# Convert image into gray scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.title('RGB image')
plt.show()

# Convert image from gray scale to RGB format
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

 #I have used OpenCV to read image. Next, I am converting the image from BGR format to GRAY format and then again converting it from gray format to RGB format. After the conversion process,I have used the Matplotlib library to print/check the image.
 
# Normalizing the image
normalized = image.astype("float32")/255.0

# Converting the image from RGB to LAB
lab = cv2.cvtColor(normalized,cv2.COLOR_RGB2LAB)

#resizing the image into 224×224 shape. changing pixels
resized = cv2.resize(lab,(224,224))

# Extracting the value of L for LAB image
L = cv2.split(resized)[0]
#The cv2.split() function splits the image into three channels, i.e. L, A, B. It is used to extract the L-channel from the LAB image by using its index number.
L -= 50
#Use the L channel as the input to the network and train the network to predict the ab channels.

#setting Input
net.setInput(cv2.dnn.blobFromImage(L))
#we are providing the L-channel as an input to our model and then predicting the “a” and “b” values from the model in the next line

# Finding the values of 'a' and 'b'
ab = net.forward()[0, :, :, :].transpose((1,2,0))

#we are resizing “a” and “b” into the shape of our input image.
ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

#the L-channel is extracted again but from the original LAB image
L = cv2.split(lab)[0]

#combined the L-channel with “a” and “b” by using Numpy to get the LAB colored image
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
# Checking the LAB image using Matplotlib 
plt.imshow(colorized)
plt.title('LAB image')
plt.show()

# Converting LAB image to BGR colored
colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
# Limits the values in array
colorized = np.clip(colorized,0,1)
# Changing the pixel intensity back to [0,255]
colorized = (255 * colorized).astype("uint8")

# Saving the image in desired path
cv2.imwrite(name2, colorized)
cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)

#saving the colorized image in the directory
cv2.imwrite(name2, colorized)
print('Successfully saved')

cv2.waitKey(0)

