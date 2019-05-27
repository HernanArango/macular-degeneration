import cv2
import numpy as np
import matplotlib.pyplot as plt
filename = "images/im0002.ppm"

def show_image(image,tittle):
    cv2.imshow(tittle, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def histogram(image):
    #channel 1 -> green
    hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    max_value = np.argmin(hist)
    print(max_value)
    prueba(image,max_value)
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def prueba(image,value):
        for x in image:
            for z in x:
                if z[1] == value:
                    z[0] = 255
                    z[2] = 255
        show_image(image,"result")

def detect_optical_disc(image):
    #Crear un kernel de '1' de 3x3 elipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    #Se aplica la transformacion: Opening
    transformacion = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)

    #Se aplica la transformacion: Dilate
    transformacion = cv2.dilate(transformacion,kernel,iterations = 1)
    histogram(transformacion)
    show_image(transformacion,"transformacion")

image = cv2.imread(filename)
b,g,r = cv2.split(image)

#blue
image[:,:,0] = 0
#green
#image[:,:,1] = 0
#red
image[:,:,2] = 0
#show_image(image,"normal")
detect_optical_disc(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
