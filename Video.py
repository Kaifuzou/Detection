import dlib
import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("egg_modle.svm")
video_path = "egg.mp4"
# Video capture source
cap = cv2.VideoCapture(video_path)

# We can look at the HOG filter we learned.  It should look like a face.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

win = dlib.image_window()
success = True
while success:

    success, image = cap.read()
    image = resize(image, width=800)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rects = detector(image)

    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(image)
    win.add_overlay(rects)