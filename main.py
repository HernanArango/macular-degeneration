import cv2
import numpy as np
import matplotlib.pyplot as plt
#filename = "images/im0009.ppm"
filename = "images/im0010.ppm"


def show_image(image,tittle):
    cv2.imshow(tittle, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def histogram(image):
    #channel 1 -> green
    hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    max_value = np.argmin(hist)
    prueba(image,max_value)
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def prueba(image,value):
    rows, cols,channel = image.shape

    #image = image[2]
    print(image[0][1])


    for i in range(0,rows):
        for j in range(0,cols):
            #print(image[i][j])
            if image[i][j][1] >= 175:
                print("entro")
                cv2.circle(image,(i,j), 20, (0,0,255), -1)
def threshold(img,t):
    rows, cols = img.shape
    new_matriz = np.zeros((rows, cols))
    #creating binary matrix
    for i in range(0,rows):
    	for j in range(0,cols):
    		if img[i][j]  >= t:
    			new_matriz[i][j] = 0
    		else:
    			new_matriz[i][j] = 1
    return new_matriz

def stretch(image):
    rows, cols, d= image.shape
    new_matriz = np.zeros((rows, cols))
    min = 0
    max = 180
    #creating binary matrix
    for i in range(0,rows):
    	for j in range(0,cols):
    		new_matriz[i][j] = ((image[i][j][1] - min) / (max - min))*255
    return new_matriz


def detect_optical_disc(image):
    show_image(image,"normal")
    image = cv2.blur(image,(25,35))

    #blue
    #image[:,:,0] = 0
    #green
    #image[:,:,1] = 0
    #red
    #image[:,:,2] = 0
    b,g,r = cv2.split(image)
    #show_image(g,"transformacionxx")

    #g = 0.2 * (np.log(1 + np.float32(g)))
    # change into uint8
    #cvuint = cv2.convertScaleAbs(g)

    #transformacion no exacta como en el paper
    new_image = cv2.convertScaleAbs(g, alpha=1.5, beta=1.5)
    #new_image = stretch(image)
    show_image(g,"resaltadoz")
    show_image(new_image,"resaltado")
    #Crear un kernel de '1' de 3x3 elipse
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    #Se aplica la transformacion: Opening
    #new_image = cv2.morphologyEx(new_image,cv2.MORPH_OPEN,kernel)

    #retval, threshold = cv2.threshold(new_image, 12, 1, cv2.THRESH_BINARY)
    # Otsu's thresholding
    #ret2,threshold = cv2.threshold(new_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = threshold(new_image,175)
    show_image(th,"transformacionzz")


    #Crear un kernel de '1' de 3x3 elipse
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    #Se aplica la transformacion: Opening
    #transformacion = cv2.morphologyEx(new_image,cv2.MORPH_OPEN,kernel)

    #Se aplica la transformacion: Dilate
    #transformacion = cv2.dilate(transformacion,kernel,iterations = 1)
    #cv2.normalize(transformacion, None, 0, 700, cv2.NORM_MINMAX)
    #histogram(transformacion)
    #show_image(transformacion,"transformacion")

image = cv2.imread(filename)
#b,g,r = cv2.split(image)

#blue
#image[:,:,0] = 0
#green
#image[:,:,1] = 0
#red
#image[:,:,2] = 0
#show_image(image,"normal")
detect_optical_disc(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
