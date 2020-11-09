# import các thư viện cần thiết
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request
import json
import cv2
import os

# define the path to the face detector
# FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascades_frontalface_default.xml".format(base_path=os.path.abspath(cv2.__file__))
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
FACE_DETECTOR_PATH = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
# hai hàm: detect và _grab_image
# detect: phát hiện là chế độ xem thực tế
@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        print("Check detector")
        print(detector)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})
    # return a JSON response
    return JsonResponse(data)


# _grab_image: là 1 chức năng nhỏ, tiện lợi để đọc ảnh từ đĩa, URL hoặc stream into OpenCV format
def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    #  otherwise, the image does not reside on disk
    # ngược lại, nếu ảnh không nằm trên disk
    else:
        # if the URL is not None, then download the image
        # nếu URL is not none, sau đó download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a Numpy array and then read it into
        # OpenCV format
        image = np.array(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


# if __name__ == '__main__':
#     print(_grab_image(url="https://pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg"))



# Create your views here.
