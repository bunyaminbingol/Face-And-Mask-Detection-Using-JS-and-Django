from django.shortcuts import render
from django.http.response import StreamingHttpResponse, HttpResponse
from MyApp.camera import MaskDetect
import json
import cv2
import base64
import numpy as np
# Create your views here.
 
 

def home(request):
    return render(request, 'MyApp/home.html')
 
def faceDetect(request):
    return render(request, 'MyApp/faceDetect.html')

def maskDetect(request):
    return render(request, 'MyApp/maskDetect.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def mask_feed(request):
    return StreamingHttpResponse(gen(MaskDetect()),
                content_type='multipart/x-mixed-replace; boundary=frame')
     
def image_process(request):
    fase_cascade = cv2.CascadeClassifier('C:/Users/Pc/CALISMALARIM/Django_web/Toyota/cascade/frontalface.xml')
    if request.method == 'POST':
        data = request.body.decode('utf-8')
        encoded_data = data.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = fase_cascade.detectMultiScale(gray, 1.3, 7)
        print(faces)
        if len(faces) > 0:
            x = int(faces[0][0])
            y = int(faces[0][1])
            w = int(faces[0][2])
            h = int(faces[0][3])
            sonuc = {
                "yuz_konum": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                },
            }
    
            return HttpResponse(json.dumps(sonuc), content_type="application/json")