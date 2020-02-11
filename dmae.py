import argparse
import cv2
import drusen

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Show proccess information", action = "store_true")
parser.add_argument("-f", "--file", help="Path to image")
parser.add_argument("-o", "--output", help="Save the Resulting Image", action = "store_true")
args = parser.parse_args()


def show_result_drusen_classification(classification_scale):
    print("Total Normal Drusen (<= 63 micron) : ",classification_scale["Normal"])
    print("Total Medium Drusen (>  63 micron and <= 125 micron) : ",classification_scale["Medium"])
    print("Total Large Drusen  (>  125 micron) : ",classification_scale["Large"])


def show_image(image, tittle):
    cv2.imshow(tittle, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image):
    if args.output:
        cv2.imwrite("result.jpg",image)

if args.verbose and args.file:
    image = cv2.imread(args.file)
    drusen_image,classification_scale = drusen.main(image,True)
    show_result_drusen_classification(classification_scale)
    show_image(drusen_image,"Drusen")
    save_image(drusen_image)
elif args.file:
    image = cv2.imread(args.file)
    drusen_image,classification_scale = drusen.main(image)
    show_result_drusen_classification(classification_scale)
    show_image(drusen_image,"Drusen")
    save_image(drusen_image)
