# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

filename = 'https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg'
cascades = 'cars.xml'

def get_image(filename):
    # Reading image
    image = Image.open(requests.get(filename,stream=True).raw)
    image = image.resize((450, 250))
    image_arr = np.array(image)
    return image, image_arr

def pre_processing(filename):
    image, image_arr = get_image(filename)

    #converting image into grayscale
    grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    Image.fromarray(grey)

    #applying GaussianBlur to remove the noise from the image
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    Image.fromarray(blur)

    #dilate image to fill the missing parts of the images
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    Image.fromarray(dilated)

    #Appling closing here
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    Image.fromarray(closing)

    return closing

def detect_model(filename, cascades):
    image, image_arr = get_image(filename) #reading the Image here
    closing = pre_processing(filename) # Pre-Processing the Image here
    car_cascade = cv2.CascadeClassifier(cascades) # Training of the images from the pretrained xml file
    cars = car_cascade.detectMultiScale(closing, 1.1, 1) #To detect multiple objects
    cnt = 0
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2) # cv2.rectangle(image, start_point, end_point, color, thickness)
        cnt += 1
    image = Image.fromarray(image_arr)
    image.save("Detect_Count_Image.jpg")
    return cnt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    count = detect_model(filename, cascades)
    print("No. of Cars Found in the Image :", count)
