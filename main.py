import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = "images/im0010.ppm"
#problemas
#filename = "images/im0014.ppm"
#filename = "images/im0023.ppm"
#filename = "images/im0015.ppm"
#filename = "images/im0037.ppm"


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

def removing_dark_pixel(image):

    rows, cols = image.shape
    new_matriz = np.zeros((rows, cols))
    #hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    average = np.average(image)
    max = np.amax(image)
    #average = max - average

    print("max",max)


    print(image[0][1])
    print("average", average)
    total = 0
    for i in range(0,rows):
        for j in range(0,cols):
            total +=image[i][j]
            if image[i][j] <= average:
                image[i][j] = 0

            #else:
                #image[i][j] = 0
    print("average2",total/(rows*cols))
    return image

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

    #show_image(image,"normal")



    #image = cv2.blur(image,(25,35))

    b,g,r = cv2.split(image)


    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    new_image = clahe.apply(g)
    show_image(new_image,"clahe")


    #new_image = cv2.blur(new_image,(1,1))
    #new_image = cv2.GaussianBlur(new_image,(1,1),0)
    #show_image(new_image,"borrosa")

    #new_image = removing_dark_pixel(new_image)
    #show_image(new_image,"removing removing_dark_pixel")
    #new_image = threshold(new_image,90)

    new_image = cv2.Canny(new_image,80,80)

    show_image(new_image,"bordes")

    new_image = cv2.dilate(new_image, None, iterations=1)
    show_image(new_image,"dilate")
    #new_image = cv2.erode(new_image, None, iterations=1)
    #show_image(new_image,"erode")
    cimg = cv2.cvtColor(new_image,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(new_image,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=30,maxRadius=90)
    if circles is not  None:

        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        show_image(cimg,'detected circles')
    else:
        print("No se encontro disco optico")



    #transformacion no exacta como en el paper
    #new_image = cv2.convertScaleAbs(g, alpha=1.5, beta=1.5)
    #show_image(new_image,"resaltado")
    #new_image = stretch(image)
    #show_image(g,"green channel")

    #new_image = threshold(new_image,175)
    #show_image(new_image,"transformacionzz")


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
