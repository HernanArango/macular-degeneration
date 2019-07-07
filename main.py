import cv2
import numpy as np
import matplotlib.pyplot as plt
import RRO_thresholding

filename = "images/im0011.ppm"
#filename = "../dataset_retinas/DRIVE/19_test.tif"
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

def sobel(image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)


    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

def detecte_circles(new_image,g):
    cimage = cv2.cvtColor(new_image,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(new_image,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=30,maxRadius=90)
    print(circles)

    if circles is not  None:

        circles = np.uint16(np.around(circles))
        max = circles[0][0]
        for i in circles[0,:]:
            # draw the outer circle
            #cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)

            #print(image[i[1]][i[0]])
            print("xxx",max[1],i[1])
            if g[i[1]][i[0]]  > g[max[1]][max[0]]:
                max = i

        cv2.circle(cimage,(max[0],max[1]),i[2],(0,250,0),2)
        cv2.circle(cimage,(max[0],max[1]),2,(0,0,200),3)



        show_image(cimage,'detected circles')
    else:
        print("No se encontro disco optico")

def detect_optical_disc(image):

    #show_image(image,"normal")

    average = np.average(image)
    print("average",average)



    b,g,r = cv2.split(image)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    new_image = clahe.apply(g)
    show_image(new_image,"clahe")


    #new_image = cv2.blur(new_image,(15,15))
    #new_image = cv2.bilateralFilter(new_image,15,75,75)
    new_image = cv2.medianBlur(new_image,15)

    #new_image = cv2.GaussianBlur(new_image,(55,55),0)
    show_image(new_image,"borrosa")

    #dark_pixel = removing_dark_pixel(new_image)
    #show_image(new_image,"removing removing_dark_pixel")
    #RRO = RRO_thresholding.RRO_thresholding()
    #new_image = RRO.calculate(new_image)

    #new_image = threshold(new_image,90)

    #new_image = cv2.Canny(new_image,80,80)
    new_image = sobel(new_image)

    #new_image = cv2.adaptiveThreshold(new_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    show_image(new_image,"bordes")



    new_image = cv2.dilate(new_image, None, iterations=1)
    #show_image(new_image,"dilate")
    #new_image = cv2.erode(new_image, None, iterations=1)
    #show_image(new_image,"erode")

    detecte_circles(new_image,g)





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
b,g,r = cv2.split(image)

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
