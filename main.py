import cv2
import numpy as np
import matplotlib.pyplot as plt
import RRO_thresholding
import copy

#filename = "images/im0010.ppm"
filename = "../dataset_retinas/DRIVE/28_training.tif"
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

def count_pixel_posibles_veins(img,x,y):
    range = 10
    total_veins = 0
    for i in range(y-range,y+range):
    	for j in range(x-range,x+range):
    		if img[i][j]  == 0:
    			total_veins += 1

    return total_veins


def count_pixel_bright(img,x,y):
    range = 10
    total_bright = 0
    for i in range(y-range,y+range):
    	for j in range(x-range,x+range):
    		total_bright += img[i][j]

    return total_bright

def sort(array,i):
    """Sort the array by using quicksort."""

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0][i]
        for x in array:
            if x[i] < pivot:
                less.append(x)
            elif x[i] == pivot:
                equal.append(x)
            elif x[i] > pivot:
                greater.append(x)
        # Don't forget to return something!
        return sort(greater,i)+equal+sort(less,i)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array



def detect_circles(edge_image,veins,g):
    cimage = cv2.cvtColor(edge_image,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(edge_image,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=30,maxRadius=90)
    print(circles)

    if circles is not  None:

        circles = np.uint16(np.around(circles))


        intensity = []
        for i in circles[0,:]:
                intensity.append([g[i[0]][i[1]],count_pixel_bright(g,i[0],i[1]),i])

        print(intensity)
        sort(intensity,0)



        tamanio = 5
        if len(intensity) < tamanio:
            tamanio = len(intensity)

        max = intensity[0]
        for i in range(0,tamanio):
            if intensity[i][1] > max[1]:
                max = intensity[i]

        #max = intensity[0]
        print("maximo",max)
        max = max[2]

        for i in circles[0,:]:
            # draw the outer circle
            #cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),3)


        cv2.circle(cimage,(max[0],max[1]),i[2],(0,250,0),2)
        cv2.circle(cimage,(max[0],max[1]),2,(0,0,200),3)



        show_image(cimage,'detected circles')

    else:
        print("No se encontro disco optico")

def detect_veins(image):
    #Kirsch's Templates

    #image = cv2.GaussianBlur(image,(5,5),0)
    h1 = np.matrix("-5 -3 -3; 5 0 -3; 5 -3 -3")
    h2 = np.matrix("-3 -3 5; -3 0 5; -3 -3 5")
    h3 = np.matrix("-3 -3 -3; 5 0 -3; 5 5 -3")
    h4 = np.matrix("-3 5 5; -3 0 5; -3 -3 -3")
    h5 = np.matrix("-3 -3 -3; -3 0 -3; 5 5 5")
    h6 = np.matrix("5 5 5; -3 0 -3; -3 -3 -3")
    h7 = np.matrix("-3 -3 -3; -3 0 5; -3 5 5")
    h8 = np.matrix("5 5 -3; 5 0 -3; -3 -3 -3")

    kernel2 = np.ones((3,3),np.float32)/25



    image1 = cv2.filter2D(image,-1,h1);
    image2 = cv2.filter2D(image,-1,h2);
    image3 = cv2.filter2D(image,-1,h3);
    image4 = cv2.filter2D(image,-1,h4);
    image5 = cv2.filter2D(image,-1,h5);
    image6 = cv2.filter2D(image,-1,h6);
    image7 = cv2.filter2D(image,-1,h7);
    image8 = cv2.filter2D(image,-1,h8);
    new_image = image1 + image2 + image3 + image4 + image5 +image6 + image7 + image8
    #show_image(new_image,"Kirsch")

    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    #show_image(gray,"gray Kirsch")
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    #gray = clahe.apply(gray)
    #show_image(new_image,"clahe Kirsch")
    veins = threshold(gray,77)

    show_image(veins," veins Kirsch")

    #new_image = cv2.GaussianBlur(veins,(3,3),0)
    #show_image(new_image,"erode")

    return veins

def detect_optical_disc(image):
    #original_image = copy.copy(image)
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
    #new_image = RRO.calculate(dark_pixel)

    #new_image = threshold(g,average)

    #new_image = cv2.Canny(g,100,100)
    #ret,new_image = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    new_image = sobel(new_image)

    #new_image = cv2.adaptiveThreshold(new_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #show_image(new_image,"bordes")


    #new_image = cv2.GaussianBlur(new_image,(3,3),0)
    #show_image(new_image,"x")
    new_image = cv2.dilate(new_image, None, iterations=1)
    show_image(new_image,"dilate")
    #new_image = cv2.erode(new_image, None, iterations=1)
    #show_image(new_image,"erode")
    veins = detect_veins(image)

    detect_circles(new_image,veins,g)





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
