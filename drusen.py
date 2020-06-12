import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import imutils
from retinex import Retinex


template1 = "./templates_od/od1.png"
template2 = "./templates_od/od2.png"
template3 = "./templates_od/od3.png"
template4 = "./templates_od/od4.png"


classification_scale = {"Normal": 0, "Medium": 0,"Large": 0}

def show_image(image, tittle):
    """
    Create a window and show a image

    Parameters:
    image (numpy array): Description of arg1
    title (string): Title to show in the window
    """
    cv2.imshow(tittle, image)


def removing_dark_pixel(image):
    """
    Delete the pixels in the image when these are minors than pixel average
    value in the image, with the objetive of reduce the pixel number and
    reduce the computer time.

    Parameters:
    image (numpy array)

    Returns:
    image (numpy array)
    """
    b, g, r = cv2.split(image)
    rows, cols,_ = image.shape
    new_matriz = np.zeros((rows, cols))
    average = np.average(r)
    max = np.amax(image)

    total = 0
    for i in range(0, rows):
        for j in range(0, cols):
            total += image[i][j][0]
            if r[i][j] <= average:
                # set zero value in the 3 channels of the pixel
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 0
    return image


def threshold(img, t):
    """
    Return a binary image. Given a value t, the algorithm transform the Pixels
    values greater than t in 1 and the values with minor value than t in 0

    Parameters:
    img (numpy array)
    t (int): threshold value

    Returns:
    image (numpy array): binary image
    """
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


def detect_roi(img, optic_disc):
    """
    with the optic disc coordinate find a nearest point near to the macula
    and create a rectangular region of interest (roi) around of
    it

    Parameters:
    img (numpy array)
    optic_disc (array(int,int)): the disc optic coordinate, x and y

    Returns:
    image (numpy array): rectangular image that cover the zone around the macula
    """
    rows, cols, _ = img.shape
    x = optic_disc[0]
    y = optic_disc[1]
    # distance of optic disc to the macula
    distance = int(cols*0.35)
    middle_image = int(cols/2)

    # detect if the macula is to the left or right
    #right
    if x > middle_image:
        x = x - distance

    #left
    else:
        x = x + distance

    # the witdh and size is 38% and 51% of the total image size repectively
    width_roi = cols * 0.387
    height_roi = rows * 0.51

    original_image = copy.copy(img)
    roi = img[y - round(height_roi/2):y + round(height_roi/2), (x - round(width_roi/2)):(x + round(width_roi/2))]

    return roi

def get_mask(img):
    """
    get the a mask of the dark zone around of the circular form in the fundus
    image, after delete it and return only the part of interest

    Parameters:
    img (numpy array)

    Returns:
    image (numpy array)
    """
    b, g, r = cv2.split(img)
    th = threshold(r, 35)
    kernel = np.ones((3, 3), np.uint8)
    # openning erosion and dilatation
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(closing, kernel, iterations=1)
    result = apply_mask(img, mask)
    return result


def apply_mask(img, mask):
    """
    delete the dark zone around of the circular form in the fundus image

    Parameters:
    img (numpy array): original image
    mask (numpy array): mask with only the black zone around the image found

    Returns:
    image (numpy array): image with the dark zone totally in zeros
    """
    rows, cols, _ = img.shape

    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i][j] == 0:
                img[i][j][0] = 0
                img[i][j][1] = 0
                img[i][j][2] = 0
    return img


def aux_template_optic_disc(template):
    """
    calculate the histogram of the three image channels (r,g,b)
    and return an array with these

    Parameters:
    template (numpy array): original image

    Returns:
    (array): return a array with the histogram by every image channel
    """
    image = cv2.imread(template)
    blue_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    return [blue_hist, green_hist, red_hist]


def template_optic_disc():
    """
    Load 4 Optic Disc templates and return a array with the average histograms
    by every channel RGB of every template.

    Parameters:
    x (int): coordinate in x
    y (int): coordinate in y

    Returns:
    (array): return a array with the average histogram by every image channel
    """
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

    return [blue_hist, green_hist, red_hist]


def hist_window(image, y, x):
    """
    given two coordinates x and y, it create a window of 80x80 pixels taken (x,y)
    like the center of the window and calculate the histogram of every channel

    Returns:
    (array): return a array with the histogram by every image channel
    """
    # window 80x80
    img_window = image[(y - 40):(y + 40), (x - 40):(x + 40)]
    blue_hist = cv2.calcHist([img_window], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([img_window], [1], None, [256], [0, 256])
    red_hist = cv2.calcHist([img_window], [2], None, [256], [0, 256])
    return [blue_hist, green_hist, red_hist]


def hist_correlation(templates_histograms, histograms_window):
    """
    use a function of correlaction for comparate the histograms of a window
    with the histogram template created previously in the function
    template_optic_disc().
    Article:
    https://jivp-eurasipjournals.springeropen.com/articles/10.1186/1687-5281-2012-19

    Parameters:
    templates_histograms (array): array with the histogram RGB
    histograms_window (array): array with the histogram RGB

    Returns:
    (int): correlation value

    """
    difference_histograms_b = np.sum(pow(templates_histograms[0] - histograms_window[0], 2))
    difference_histograms_g = np.sum(pow(templates_histograms[1] - histograms_window[1], 2))
    difference_histograms_r = np.sum(pow(templates_histograms[2] - histograms_window[2], 2))

    Cb = 1 / (1 + difference_histograms_b)
    Cg = 1 / (1 + difference_histograms_g)
    Cr = 1 / (1 + difference_histograms_r)
    Tb = 1
    Tg = 2
    Tr = 0.5

    C = (Tr * Cr) + (Tb * Cb) + (Tg * Cg)

    return C


def threshold_color(otsu_img,img):
    """
    the function make a thresholding over a binary image and delete the pixel
    that in the original image have a minor value to the average of the original
    image

    Parameters:
    otsu_img (numpy array): binary image
    img (numpy array): original image

    Returns:
    (numpy array): new binary image

    """
    b, g, r = cv2.split(img)
    average = np.average(g)
    rows, cols = otsu_img.shape

    for i in range(0, rows):
        for j in range(0, cols):

            if otsu_img[i][j] != 0:
                if g[i][j] < average:
                    otsu_img[i][j] = 0

    return otsu_img

def non_uniform_ilumination_correction(img):
    """
    the function normalize the ilumination in fundus image, the method is
    explained in
    G. L. Yang Gijoo Wang Shinn-Wen, «Algorithm for detecting micro-aneurysms in
    low-resolution color retinal images», 2001.

    Parameters:
    img (numpy array): original image

    Returns:
    (numpy array): new image with image correction
    """
    b, g, r = cv2.split(img)

    fundus = cv2.medianBlur(g, 71)
    new_image = (g/fundus)
    new_image = (new_image*55).astype(np.uint8)
    new_image = cv2.GaussianBlur(new_image,(1,1),0,0,cv2.BORDER_DEFAULT)
    return new_image


def detect_drusen(img):
    """
    detect the drusen in the image and draw their contours, use a strategy based
    in J. F. André Mora Pedro Vieira, «Drusen Deposits on Retina Images: Detection and
    Modeling», 2014.

    Parameters:
    img (numpy array): the roi in the image

    Returns:
    (numpy array): new image with the contours of the drusen

    """
    b, g, r = cv2.split(img)

    g = non_uniform_ilumination_correction(img)
    fundus = cv2.medianBlur(g, 71)
    g = cv2.GaussianBlur(g,(15,15),0,0,cv2.BORDER_DEFAULT)
    x = (fundus/g)*1.09
    new_image = (x*255).astype(np.uint8)
    # threshold Otsu
    ret, otsu_img = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU)
    veins = detect_veins(g)
    # evite division by zero
    edge_map = np.zeros_like(veins)
    non_zero = veins != 0
    edge_map[non_zero] = otsu_img[non_zero]/veins[non_zero]
    otsu_img = edge_map

    ret, otsu_img = cv2.threshold(otsu_img, 0, 255, cv2.THRESH_OTSU)
    otsu_img = threshold_color(otsu_img,img)

    #using erotion and dilation for evite count more contours by error
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    otsu_img = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(otsu_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        size_drusen(rect[1])

    return img




def size_drusen(dimensions):
    """
    it make a size clasification every drusen based in the basic classification
    of AMD
    M. Paul, «Age-related macular degeneration», Lancet (London, England), vol. 392,
    pp. 1147-1159, 2018, issn: 1474-547X. doi: 10.1016/S0140-6736(18)31550-2 .

    Parameters:
    dimension (array): with 2 values corresponding to the side of the rectangle

    """
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


def detect_optical_disc(image):
    """
    to find the optic disc in a fundus image and return a point with the aproximate
    ubication of it

    Parameters:
    image (numpy array):

    Returns:
    (array(int,int)): array with the coordinate x,y of the point of the OD
    """
    start = time.time()
    original_image = copy.copy(image)
    templates_histograms = template_optic_disc()
    b, g, r = cv2.split(image)
    image = cv2.medianBlur(image, 5)
    rows, cols, _ = image.shape
    correlations = []
    new_matriz = np.zeros((rows, cols))

    for i in range(200, rows-230):
        for j in range(0, cols):

            if image[i][j][0] != 0:
                window_histogram = hist_window(image, i, j)
                new_matriz[i][j] = hist_correlation(templates_histograms, window_histogram)


    max = new_matriz[0][0]
    for i in range(0, rows):
        for j in range(0, cols):
            if new_matriz[i][j] > max:
                max = new_matriz[i][j]


    image_threshold = threshold(new_matriz, max * 0.7)
    end = time.time()

    kernel = np.ones((3, 3), np.uint8)
    # opening = erosion followed by dilation
    gradient = cv2.morphologyEx(image_threshold, cv2.MORPH_GRADIENT, kernel)

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


    y = int(y_min + ((y_max - y_min) / 2))
    x = int(x_min + ((x_max - x_min) / 2))

    return [x,y]


def detect_veins(g):
    """
    remove the veins in a fundus image, this algorithm is not the most robust for
    the elimination of veins but work very well in the presence of drusen, based
    in B. N. Dash Jyotiprava, «A thresholding based technique to extract retinal blood
    vessels from fundus images», Future Computing and Informatics Journal, vol. 2, n. o 2,
    pp. 103-109, 2017, issn: 2314-7288. doi: 10.1016/j.fcij.2017.10.001 .

    Parameters:
    g (numpy array): channel green of a image

    Returns:
    (numpy array): binary image with the veins in white and the background in black

    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    gray = clahe.apply(g)
    image_filtered = cv2.GaussianBlur(gray,(21,21),0)
    th2 = cv2.adaptiveThreshold(image_filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    dilate = cv2.dilate(th2, kernel)

    return dilate


def change_resolution(img):
    img = imutils.resize(img, width=700)
    return img


def main(image, debug = False):

    original_image = copy.copy(image)
    if debug:
        print("# Changing Resolution")
    image = change_resolution(image)

    cols_original, rows_original, _ = original_image.shape
    cols_modified, rows_modified, _ = image.shape
    # Get the original ratio
    Rx = (rows_original/rows_modified)
    Ry = (cols_original/cols_modified)

    if debug:
        print("# Removing Dark Pixels")
    image = removing_dark_pixel(image)
    if debug:
        print("# Detecting Optical Disc")
    x,y = detect_optical_disc(image)

    if debug:
        print("# Calculating ROI")
    roi = detect_roi(original_image, [round(x*Rx), round(y*Ry)])
    if debug:
        print("# Segmenting Drusen")

    drusen = detect_drusen(roi)


    return [drusen,classification_scale]
