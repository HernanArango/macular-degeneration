import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.cluster import KMeans
import time
import imutils


template1 = "./templates_od/od1.png"
template2 = "./templates_od/od2.png"
template3 = "./templates_od/od3.png"
template4 = "./templates_od/od4.png"



classification_scale = {"Normal": 0, "Medium": 0,"Large": 0}

def show_image(image, tittle):
    cv2.imshow(tittle, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def histogram(image):
    # channel 1 -> green
    hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    max_value = np.argmin(hist)

    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def removing_dark_pixel(image):
    b, g, r = cv2.split(image)
    rows, cols,_ = image.shape
    new_matriz = np.zeros((rows, cols))
    # hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    average = np.average(r)
    max = np.amax(image)
    # average = max - average

    # print("max", max)

    # print(image[0][1])
    # print("average", average)
    total = 0
    for i in range(0, rows):
        for j in range(0, cols):
            total += image[i][j][0]
            if r[i][j] <= average:
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 0

            # else:
            # image[i][j] = 0
    # print("average2", total / (rows * cols))
    return image


def threshold(img, t):
    rows, cols = img.shape
    new_matriz = np.zeros((rows, cols))
    # creating binary matrix
    for i in range(0, rows):
        for j in range(0, cols):
            if img[i][j] == t:
                new_matriz[i][j] = 1
    return new_matriz


def threshold2(img, t):
    rows, cols = img.shape
    new_matriz = np.zeros((rows, cols))
    # creating binary matrix
    for i in range(0, rows):
        for j in range(0, cols):
            if img[i][j] >= t:
                new_matriz[i][j] = 1
            else:
                new_matriz[i][j] = 0
    return new_matriz


def stretch(image):
    rows, cols, d = image.shape
    new_matriz = np.zeros((rows, cols))
    min = 0
    max = 180
    # creating binary matrix
    for i in range(0, rows):
        for j in range(0, cols):
            new_matriz[i][j] = ((image[i][j][1] - min) / (max - min)) * 255
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


def sort(array, i):
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
        return sort(greater, i) + equal + sort(less, i)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array


def count_pixel_veins(veins, x, y):
    total_veins = 0
    for i in range(y - 50, y + 50):
        for j in (x - 50, x + 50):
            if veins[i][j] == 0:
                total_veins += 1

    return total_veins


def count_pixel_bright(img, x, y):
    total_veins = 0
    for i in range(y - 10, y + 10):
        for j in (x - 10, x + 10):
            total_veins += img[i][j]

    return total_veins


def detect_circles(edge_image, veins, g):
    cimage = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(edge_image, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=30, maxRadius=90)
    # print(circles)

    if circles is not None:

        circles = np.uint16(np.around(circles))

        intensity = []
        for i in circles[0, :]:
            intensity.append([count_pixel_bright(g, i[0], i[1]), count_pixel_veins(veins, i[0], i[1]), i])

        # count_pixel_posibles_veins(veins,i[0],i[1])

        intensity = sort(intensity, 0)
        # print("intensity", intensity)

        tamanio = 5  # we choose only the 5 first values
        if len(intensity) < tamanio:
            tamanio = len(intensity)

        max = intensity[0]
        for i in range(0, tamanio):
            if intensity[i][0] > max[0]:
                max = intensity[i]

        # select only the data of the circle (x y radius)
        max = max[2]
        # print("maximo", max)

        for i in circles[0, :]:
            # draw the outer circle
            # cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimage, (i[0], i[1]), 2, (0, 0, 255), 3)

        for i in range(0, tamanio):
            cv2.circle(cimage, (intensity[i][2][0], intensity[i][2][1]), 2, (255, 0, 0), 3)

        cv2.circle(cimage, (max[0], max[1]), max[2], (0, 250, 0), 2)
        cv2.circle(cimage, (max[0], max[1]), 2, (255, 0, 0), 3)

        # show_image(cimage, 'detected circles')

    else:
        print("No se encontro disco optico")

    return max


def detect_veins(image):
    b, g, r = cv2.split(image)
    # show_image(g,"verde")
    # Crear un kernel de '1' de 3x3 elipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    # Se aplica la transformacion: Opening
    transformacion = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    # show_image(transformacion,"transformacion")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    tophat = cv2.morphologyEx(transformacion, cv2.MORPH_BLACKHAT, kernel)
    # show_image(tophat,"transformacion2")

    Z = tophat.reshape((-1, 1))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # print("center", center)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((tophat.shape))
    res2 = threshold(res2, center[0])
    # show_image(res2, "kmeans")

    return res2

    """
    X = tophat.reshape((-1, 1))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    # Assign each value to the nearest centroid and
    # reshape it to the original image shape
    input_image_compressed = np.choose(labels, centroids).reshape(tophat.shape)
    show_image(input_image_compressed,"threshold")
    """


def detect_veins2(image):
    # Kirsch's Templates

    # image = cv2.GaussianBlur(image,(5,5),0)
    h1 = np.matrix("-5 -3 -3; 5 0 -3; 5 -3 -3")
    h2 = np.matrix("-3 -3 5; -3 0 5; -3 -3 5")
    h3 = np.matrix("-3 -3 -3; 5 0 -3; 5 5 -3")
    h4 = np.matrix("-3 5 5; -3 0 5; -3 -3 -3")
    h5 = np.matrix("-3 -3 -3; -3 0 -3; 5 5 5")
    h6 = np.matrix("5 5 5; -3 0 -3; -3 -3 -3")
    h7 = np.matrix("-3 -3 -3; -3 0 5; -3 5 5")
    h8 = np.matrix("5 5 -3; 5 0 -3; -3 -3 -3")

    kernel2 = np.ones((3, 3), np.float32) / 25

    image1 = cv2.filter2D(image, -1, h1);
    image2 = cv2.filter2D(image, -1, h2);
    image3 = cv2.filter2D(image, -1, h3);
    image4 = cv2.filter2D(image, -1, h4);
    image5 = cv2.filter2D(image, -1, h5);
    image6 = cv2.filter2D(image, -1, h6);
    image7 = cv2.filter2D(image, -1, h7);
    image8 = cv2.filter2D(image, -1, h8);
    new_image = image1 + image2 + image3 + image4 + image5 + image6 + image7 + image8
    # show_image(new_image,"Kirsch")

    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # show_image(gray,"gray Kirsch")
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    # gray = clahe.apply(gray)
    # show_image(new_image,"clahe Kirsch")
    veins = threshold(gray, 77)

    # show_image(veins, " veins Kirsch")

    # new_image = cv2.GaussianBlur(veins,(3,3),0)
    # show_image(new_image,"erode")

    return veins


def detect_roi(img, optic_disc):
    rows, cols, _ = img.shape
    # print("rows",rows,"cols",cols)
    x = optic_disc[0]
    y = optic_disc[1]
    # distance of optic disc to the macula
    distance = int(cols*0.35)
    middle_image = int(cols/2)

    cv2.circle(img, (x, y), 2, (0, 255, 0), 3)
    translate_rect = 0
    # detect if the macula is to the left or right
    #right
    if x > middle_image:

        x = x - distance
        translate_rect = 0
    #left
    else:

        x = x + distance
        translate_rect = 70

    #y = optic_disc[1] + 30
    # print(x, y)
    # print("oprio", optic_disc)
    #cv2.circle(img, (x, y), 35, (0, 255, 0), 100)
    # show_image(img,"puntos")


    # cv2.rectangle(img, (x - 500 +translate_rect, y - 450), (x + 500 + translate_rect, y + 550), (0, 255, 0), 3)


    # print(x - 200+translate_rect)
    #roi = img[y - 150:y + 150, (x - 200)+translate_rect:(x + 200)+translate_rect]
    roi = img[y - 450:y + 550, (x - 500)+translate_rect:(x + 500)+translate_rect]

    # show_image(imutils.resize(img, width=700), 'pequena macula')

    return roi

def get_mask(img):
    b, g, r = cv2.split(img)
    th = threshold2(r, 35)
    kernel = np.ones((3, 3), np.uint8)
    # openning erosion and dilatation
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(closing, kernel, iterations=1)
    result = apply_mask(img, mask)
    return result


def apply_mask(img, mask):
    rows, cols, _ = img.shape

    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i][j] == 0:
                img[i][j][0] = 0
                img[i][j][1] = 0
                img[i][j][2] = 0
    return img


def aux_template_optic_disc(template):
    image = cv2.imread(template)
    # image_filtered = cv2.medianBlur(image, 5)
    # show_image(image_filtered,"verde")
    blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])
    """
    plt.plot(blue_hist)
    plt.plot(green_hist)
    plt.plot(red_hist)
    plt.xlim([0, 256])
    plt.show()
    """

    return [blue_hist, green_hist, red_hist]


def template_optic_disc():
    print("## Creating histogram template")
    hist1 = aux_template_optic_disc(template1)
    hist2 = aux_template_optic_disc(template2)
    hist3 = aux_template_optic_disc(template3)
    hist4 = aux_template_optic_disc(template4)

    blue_hist = []
    for i in range(0, 256):
        blue_hist.append((hist1[0][i] + hist2[0][i] + hist3[0][i] + hist4[0][i]) / 4)

    green_hist = []
    for i in range(0, 256):
        green_hist.append((hist1[1][i] + hist2[1][i] + hist3[1][i] + hist4[1][i]) / 4)

    red_hist = []
    for i in range(0, 256):
        red_hist.append((hist1[2][i] + hist2[2][i] + hist3[2][i] + hist4[2][i]) / 4)
    """
    plt.plot(blue_hist)
    plt.plot(green_hist)
    plt.plot(red_hist)
    plt.xlim([0, 256])
    plt.show()
    """
    return [blue_hist, green_hist, red_hist]


def hist_window(image, y, x):
    img_window = image[(y - 40):(y + 40), (x - 40):(x + 40)]

    # img_window = np.asarray(img_window)
    # show_image(img_window,"80x80")
    blue_hist = cv2.calcHist([img_window], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([img_window], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([img_window], [2], None, [256], [0, 256])
    return [blue_hist, green_hist, red_hist]


def hist_correlation(templates_histograms, histograms_window):
    difference_histograms_b = np.sum(pow(templates_histograms[0] - histograms_window[0], 2))
    difference_histograms_g = np.sum(pow(templates_histograms[1] - histograms_window[1], 2))
    difference_histograms_r = np.sum(pow(templates_histograms[2] - histograms_window[2], 2))

    Cb = 1 / (1 + difference_histograms_b)
    Cg = 1 / (1 + difference_histograms_g)
    Cr = 1 / (1 + difference_histograms_r)
    Tb = 1
    Tg = 2
    Tr = 0.5

    # print(diffence_histograms_b)
    C = (Tr * Cr) + (Tb * Cb) + (Tg * Cg)
    # print("C",C[0])
    return C


def detect_drusas(img):
    # show_image(img, "1 normal")

    b, g, r = cv2.split(img)
    # show_image(g, "green")
    fundus = cv2.medianBlur(g, 41)
    #show_image(g,"mediana")
    #fundus = cv2.GaussianBlur(g,(51,51),100.0,100.0,cv2.BORDER_DEFAULT)
    #fundus = cv2.GaussianBlur(g,(91,91),100.0,100.0,cv2.BORDER_DEFAULT)
    # show_image(fundus,"gaussian blur")
    g = cv2.GaussianBlur(g,(7,7),0,0,cv2.BORDER_DEFAULT)
    # show_image(g,"g2")
    #x = (fundus/g)*1.09
    x = (fundus/g)*1.09
    new_image = (x*255).astype(np.uint8)
    # show_image(new_image,"resultado")

    # threshold
    ret, otsu_img = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU)
    # show_image(otsu_img, "otsu")


    x = detect_veins3(img)
    otsu_img = (otsu_img/x).astype(np.uint8)
    # show_image(otsu_img,"aaaa")


    contours,_ = cv2.findContours(otsu_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # dibujar los contornos
    total = 0
    for c in contours:
        cv2.drawContours(img, [c], 0, (0, 255, 0), 2, cv2.LINE_AA)
        total = total + 1

        momentos = cv2.moments(c)
        if momentos['m10']== 0 or momentos['m00']==0:
            cx = 0
        else:
            cx = int(momentos['m10']/momentos['m00'])


        if momentos['m01']== 0 or momentos['m00']==0:
            cy = 0
        else:
            cy = int(momentos['m01']/momentos['m00'])

        #cv2.circle(img,(cx, cy), 3, (0,0,255), -1)

        #size_drusas([cx,cy],momentos)
        #break

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        im = cv2.drawContours(img,[box],0,(255,0,0),2)
        size_drusas(rect[1])
        """
        cx = int(momentos['m10']/momentos['m00'])
        cy = int(momentos['m01']/momentos['m00'])
        #Dibujar el centro
        cv2.circle(img,(cx, cy), 3, (0,0,255), -1)
        """

    #print("total drusas",total)

    return img
    # show_image(img,"contornos")


def size_drusas(dimensions):
    diameter = 0
    #bigest side
    if dimensions[0] > dimensions[1]:
        diameter = dimensions[0]
    else:
        diameter = dimensions[1]

    micron = 3.4
    # transform diameter in pixel to micron -> 1px = 3.4 micron
    diameter = diameter * micron


    #normal --->   <= 63 micron
    if diameter <= 63:
        classification_scale["Normal"] += 1
    #Early AMD --->  Medium Drusen > 63 micron and <= 125 miron
    elif diameter > 63 and diameter <= 125:
        classification_scale["Medium"] += 1
    #Intermetiate AMD --> Large Drusen > 125 micron
    else:
        classification_scale["Large"] += 1


def change_background(img):
    rows, cols = img.shape
    new_matriz = np.zeros((rows, cols))
    # creating binary matrix
    for i in range(0, rows):
        for j in range(0, cols):
            #print(img[i][j])
            if img[i][j] <= 10:
                img[i][j] = 255

    return img

def detect_optical_disc(image):
    start = time.time()
    # print("mask")
    original_image = copy.copy(image)
    #image = get_mask(image)
    # show_image(image, "normal")
    # print("template")
    templates_histograms = template_optic_disc()

    # average = np.average(image)
    # print("average",average)

    b, g, r = cv2.split(image)
    image = cv2.medianBlur(image, 5)

    rows, cols, _ = image.shape

    # window_histogram = hist_window(image,500,200)
    # hist_correlation(templates_histograms,window_histogram)

    correlations = []
    new_matriz = np.zeros((rows, cols))
    # print(rows, cols)
    for i in range(200, rows-230):
        # print(i)
        for j in range(0, cols):


            if image[i][j][0] != 0:
                window_histogram = hist_window(image, i, j)
                new_matriz[i][j] = hist_correlation(templates_histograms, window_histogram)

    # print("calculando el max")
    max = new_matriz[0][0]
    for i in range(0, rows):
        for j in range(0, cols):
            if new_matriz[i][j] > max:
                max = new_matriz[i][j]

    # print("max", max)
    # print("threshold")
    image_threshold = threshold2(new_matriz, max * 0.7)
    end = time.time()
    # print(end - start)
    kernel = np.ones((3, 3), np.uint8)
    # show_image(image_threshold, "image threshold")
    # opening = erosion followed by dilation
    gradient = cv2.morphologyEx(image_threshold, cv2.MORPH_GRADIENT, kernel)
    # show_image(gradient, "gradient")

    # calulating the center of optic disc
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0
    for i in range(0, rows):
        for j in range(0, cols):
            # only enter one time
            if gradient[i][j] == 1 and y_max == 0:
                # initial definition only one time enter here
                y_min = i
                x_min = j
                x_max = j
            if gradient[i][j] == 1:
                y_max = i

                if x_min > j:
                    x_min = j
                elif x_max < j:
                    x_max = j

    # print(x_min, x_max, y_min, y_max)
    y = int(y_min + ((y_max - y_min) / 2))
    x = int(x_min + ((x_max - x_min) / 2))

    # x = 142
    # y = 227
    #cv2.circle(image, (x, y), 2, (255, 0, 0), 3)
    ## show_image(image, 'circles')

    return [x,y]


def detect_veins3(img):
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    gray = clahe.apply(g)
    #image_filtered = cv2.medianBlur(gray, 1)
    image_filtered = cv2.GaussianBlur(gray,(21,21),0)
    th2 = cv2.adaptiveThreshold(image_filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    # show_image(th2,"hola")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilate = cv2.dilate(th2, kernel)
    #dilate = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    # show_image(dilate,"hola2")
    """
    final = g/dilate
    show_image(final,"final")
    """
    return dilate #final

def change_resolution(img):
    """
    #percent by which the image is resized
    scale_percent = 50
    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(img, dsize)

    cv2.imwrite('/home/hernan/Escritorio/prueba.png',output)
    """
    img = imutils.resize(img, width=700)
    return img

def main(image):

    original_image = copy.copy(image)

    #b, g, r = cv2.split(image)

    # blue
    # image[:,:,0] = 0
    # green
    # image[:,:,1] = 0
    # red
    # image[:,:,2] = 0
    image = change_resolution(image)

    cols_original, rows_original, _ = original_image.shape
    cols_modified, rows_modified, _ = image.shape
    # Get the original ratio
    Rx = (rows_original/rows_modified)
    Ry = (cols_original/cols_modified)

    

    # show_image(image,"normal")
    print("# Removing dark pixel")
    image = removing_dark_pixel(image)
    # show_image(image,"dark")
    print("# Detecting Optical Disc")
    x,y = detect_optical_disc(image)
    cv2.circle(image, (x, y), 2, (255, 0, 0), 3)
    # show_image(image, 'circles')

    #cv2.circle(original_image, (round(x*Rx), round(y*Ry)), 2, (255, 0, 0), 3)
    #cv2.circle(original_image, (round(x*Rx), round(y*Ry)), 35, (255, 0, 0), 100)

    # show_image(original_image, 'circles2')
    # print("OD pequenio",x,y)
    # print("OD grande",round(x*Rx),round(y*Ry))

    print("# Calculating ROI")
    roi = detect_roi(original_image, [round(x*Rx), round(y*Ry)])
    # show_image(roi,"roi")
    print("# Segmenting Drusen")
    drusas = detect_drusas(roi)
    show_image(drusas,"Drusas")
    print("Total Normal Drusen (<= 63 micron) : ",classification_scale["Normal"])
    print("Total Medium Drusen (>  63 micron and <= 125 micron) : ",classification_scale["Medium"])
    print("Total Large Drusen  (>  125 micron) : ",classification_scale["Large"])

    # print(classification_scale)

    #detect_veins(x)
    #detect_drusas(x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return drusas