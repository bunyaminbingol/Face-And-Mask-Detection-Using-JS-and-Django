# Face-And-Mask-Detection-Using-JS-and-Django

Hi! 
The goal here is to run face and mask detection with the **Django** framework using the **JavaScript (JS)** and **Python** programming languages. The images taken from the camera opened in **JS** are sent to Python in base64 format via the api, and it converts the base64 in Python and places it in the model and performs the detection. It takes the detected coordinates and adds them to a dictionary structure (JSON) and sends that structure to **JS**. Creates a canvas next to the camera area opened in **JS** and draws face coordinates on that canvas from Python. This way, you can run any real-time detection model on **Django** using **Python** and **JS**.


# How to RUN
* First you need to start **python**
> You need to run `python manage.py runserver`
* In the navbar section, you can select both options and perform the test.

#

**You can use this work in all the modules you create.**
