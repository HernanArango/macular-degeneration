import argparse
import cv2
import drusen

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Show proccess information")
parser.add_argument("-f", "--file", help="Path to image")
args = parser.parse_args()

if args.verbose:
    print ("depuración activada!!!")
if args.file:
	image = cv2.imread(args.file)
	drusen.main(image)
