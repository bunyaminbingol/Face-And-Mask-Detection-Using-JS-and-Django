from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
import time

face_detection_webcam = cv2.CascadeClassifier(os.path.join(
    settings.BASE_DIR, 'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))

# modeli yüklüyoruz
prototxtPath = "C:/Users/Pc/CALISMALARIM/Django_web/Mask-Detection-Using-JS-and-Django/face_detector/deploy.prototxt"
weightsPath = "C:/Users/Pc/CALISMALARIM/Django_web/Mask-Detection-Using-JS-and-Django/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("C:/Users/Pc/CALISMALARIM/Django_web/Mask-Detection-Using-JS-and-Django/face_detector/mask_detector.model")

class MaskDetect(object):
    def __init__(self):
        self.vs = VideoStream(src=0).start()
        self.prev_frame_time= 0
        self.new_frame_time = 0
        

    def __del__(self):
        cv2.destroyAllWindows()

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        self.new_frame_time=time.time()
        fps = 1/(self.new_frame_time-self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        print('FPS : %.2f  ' % fps)

        # çerçevenin boyutlarını alın ve ondan bir blob oluşturuyoruz
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # blobu ağ üzerinden geçirin ve yüz algılamalarını alıyoruz
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # yüzler listemizi, bunlara karşılık gelen konumları ve yüz maskesi ağımızdan tahminlerin listesini başlatıyoruz
        faces = []
        locs = []
        preds = []

        # detection üzerinden döngüyü başlatıyoruz
        for i in range(0, detections.shape[2]):
            # algılama olasılığı
            confidence = detections[0, 0, i, 2]

            # güvenin minimum güvenden daha büyük olduğu sürece zayıf algılamaları filtreliyoruz
            if confidence > 0.5:
                # nesne için sınırlayıcı kutunun (x, y)-koordinatlarını hesaplıyoruz
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # sınırlayıcı kutuların çerçevenin boyutları dahilinde olduğunu kontrol ediyoruz
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # yüz ROI'sini çıkarıyoruz, BGR'den RGB kanal sıralamasına dönüştürüyoruz, 224x224'e yeniden boyutlandırıp işliyoruz
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # yüz ve sınırlayıcı kutuları ilgili listelerine ekliyoruz
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # bir yüz algılandıysa tahmin ediyoruz
        if len(faces) > 0:
            # daha hızlı çıkarım için yukarıdaki "for" döngüsündeki tek tek tahminler yerine tüm yüzler üzerinde aynı anda toplu tahminler yapıyoruz
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)  # yüzlerin konumları ve maskelerin tahminleri

    def get_frame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=650)
        frame = cv2.flip(frame, 1)
   
        # çerçevedeki yüzleri algılayın ve yüz maskesi takıp takmadıklarını belirliyoruz
        (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

        # algılanan yüz konumları ve bunlara karşılık gelen konumlar üzerinde döngü
        for (box, pred) in zip(locs, preds):
            # sınırlayıcı kutuyu ve tahminleri açıyoruz
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # sınırlayıcı kutuyu çizmek ve metni yazdırmakş için kullanacağımız sınıf etiketini ve rengi belirliyoruz
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # tahmin oranını yazdırıyoruz
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # etiket ve sınırlayıcı kutu çıktısını görüntülüyoruz
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()