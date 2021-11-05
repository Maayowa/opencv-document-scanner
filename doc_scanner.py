import numpy as np
import cv2
import utils
import argparse
from skimage.filters import threshold_local


# construct the argument parse and parse the arguments

parser = argparse.ArgumentParser(description="Enter args to blur or scan a document",
                                    epilog="\u00A9 Oluwabunmi Iwakin")

# Adding arguments #path, obj, tech, outpath
parser.add_argument("impath", type=str, help=": path to input image")
parser.add_argument('obj',type=str, help=": objective. choice: 'scan', 'blur' ")
parser.add_argument("-s", "--brush_size", type=int, default= 10, help=": size of blurring brush")
parser.add_argument("-t", "--technique", type=str, default= 'auto', 
                    help=": techinque for detecting document. choice: 'auto', 'pts', 'rect'")
parser.add_argument("-p", "--outpath", type=str, default= 'scandoc.jpg',
                    help=": path to save image to. default = 'scandoc.jpg'")


def get_edge(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)

    return edges


def get_auto_contours(img):

    edge = get_edge(img)
    contours, _ = cv2.findContours(
        edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area for each detected contour.
    #  Assumes the largest is a rectangle or paper.
    # Selects the 5 contours with the largest areas
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        # Draw Approzimate contour shape
        perimeter = cv2.arcLength(contour, True)
        eps = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)

        # Searching for single rect image
        # If n approx contours is 4 then we can say contour is a rectangle
        if len(approx) == 4:
            opt_contour = approx
            break

    return opt_contour


def get_points_contours(img):

    points = []

    def mouse_draw(event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    font = cv2.FONT_HERSHEY_COMPLEX
    text = "Press 'r' to restart; Press 'q' to continue"
    cv2.putText(img, text, (15, 15), font, 0.5, (255, 72, 155), 1)

    temp = img.copy()
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", mouse_draw)

    while True:

        close = False

        if len(points) == 4:
            points = utils.order_coords(points)
            points = points.astype(int).tolist()
            img = temp.copy()
            close = True

        cv2.polylines(img=img,
                      pts=[np.array(points)],
                      isClosed=close,
                      color=(0, 0, 255),
                      thickness=2)
        cv2.imshow('frame', img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            img = temp.copy()
            points = []

        if key == ord("q"):
            img = temp.copy()
            break

    cv2.destroyAllWindows()

    return points


def get_rect_contours(img):

    orig = img.copy()

    while True:

        font = cv2.FONT_HERSHEY_COMPLEX
        text1 = "Press 'Enter' twice to save and exit"
        cv2.putText(img, text1, (50, 50), font, 0.5, (255, 72, 155), 1)

        r = cv2.selectROI("Image", img, False, False)  # as x,y,w,h

        cv2.imshow('image', r)

        x, y, h, w = r
        contour = [(x, y), (x, y + h), (x + w, y),  (x + w, y + h)]

        if cv2.waitKey(0):
            break

        img = orig.copy()

    cv2.destroyAllWindows()

    return contour


def get_scan(img, contour):

    warped = utils.corner_transform(img, contour)

    # cropped image to scanned greyscale format
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = threshold_local(warped, 11, offset=10)  # neighborhood mask
    warped = (warped > thresh).astype("uint8") * 255

    return warped


def blurrer(img, size, impath):
    action = False

    # Mask selected region
    def drawMask(x, y, size=size):

        # Sample processing
        m = int(x / size * size)
        n = int(y / size * size)

        for i in range(int(size)):
            for j in range(int(size)):
                img[m + i][n + j] = img[m][n]

    def drawblur(event, x, y, flags, param):

        global action

        # Left click to start action
        if event == cv2.EVENT_LBUTTONDOWN:
            action = True

        # Left click and move
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
            if action:
                drawMask(y, x)
            # Release left mouse button to end operation
            elif event == cv2.EVENT_LBUTTONUP:
                action = False

    # Open Image Window
    cv2.namedWindow('Image')
    # Start up mouse activity with the draw function
    cv2.setMouseCallback('Image', drawblur)

    font = cv2.FONT_HERSHEY_COMPLEX
    text = "Press 'r' to restart; Press 's' to save and exit"
    cv2.putText(img, text, (15, 15), font, 0.5, (255, 72, 155), 1)

    temp = img.copy()

    # Keep window and mouse activity running
    while(1):
        cv2.imshow('Image', img)

        # escape key to log out
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(impath, img)
            break

        elif key == ord("r"):
            img = temp.copy()

    # Close Windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    args = parser.parse_args()

    path = args.impath
    obj = args.obj
    bsize = args.brush_size
    tech = args.technique
    outpath = args.outpath

    image = cv2.imread(path)
    orig = image.copy()

    if obj == 'scan':

        if tech == "auto":

            # resize to make faster computation
            ratio = image.shape[0] / 500.0
            image = utils.resize(image, height=500)
            contour = get_auto_contours(image)
            contour = contour.reshape(4, 2) * ratio

        elif tech == "pts":
            contour = get_points_contours(image)

        elif tech == "rect":
            contour = get_rect_contours(image)

        out = get_scan(orig.copy(), contour)
        cv2.imwrite(outpath, out)

    elif obj == 'blur':
        blurrer(image, bsize, outpath)
