from logging import debug
from cvlib import object_detection
from flask import Flask
import flask
from flask_restful import Api, Resource
import cv2
import numpy as np
import cvlib as cv
import urllib.request
import time


app = Flask(__name__)
api = Api(app)

class ProductImageProcess(Resource):
    def post(self):
        imagefile = flask.request.form.get('image')
        isApproved = False

        imagefile = str(imagefile)
 
        media = "?alt=media"
        print(imagefile + media)

        req = urllib.request.urlopen(imagefile + media)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        laplacian_var = cv2.Laplacian(img, cv2.COLOR_BGR2GRAY).var()

        # old code
        # box,label,count = cv.detect_common_objects(img)
        # objectCount = str(len(label))

        objectCount = self.objectCount(img)
        # objectCount = 1

        if int(objectCount) >= 1 and laplacian_var > 30:
            isApproved = True

        return {"is_approved":isApproved, "image_blur":laplacian_var, "object_count":objectCount}
    
    def objectCount(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        # Use minSize because for not 
        # bothering with extra-small 
        # dots that would look like STOP signs
        stop_data = cv2.CascadeClassifier('cascade.xml')
        
        found = stop_data.detectMultiScale(img_gray, 
                                        minSize =(20, 20))
        
        # Don't do anything if there's 
        # no sign
        amount_found = len(found)
        return amount_found

class GenderImageProcess(Resource):

    def getFaceBox(self, net, frame, conf_threshold=0.7):
        print("masuk getFaceBox")
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes

   
    def age_gender_detector(self, frame):
        print("masuk age gender detector")
        # Read frame
        faceProto = "modelNweight/opencv_face_detector.pbtxt"
        faceModel = "modelNweight/opencv_face_detector_uint8.pb"

        ageProto = "modelNweight/age_deploy.prototxt"
        ageModel = "modelNweight/age_net.caffemodel"

        genderProto = "modelNweight/gender_deploy.prototxt"
        genderModel = "modelNweight/gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        # Load network
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        faceNet = cv2.dnn.readNet(faceModel, faceProto)

        padding = 20

        results = []

        t = time.time()
        frameFace, bboxes = self.getFaceBox( faceNet, frame)
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            result = {'age': age, 'gender': gender}
            results.append(result)

            label = "{},{}".format(gender, age) 
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return results

    def post(self):
        imagefile = flask.request.files.get('imagefile', '').read()
 
        npimg = np.fromstring(imagefile, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        output = self.age_gender_detector(img)

        return output
        # return {"age":12, "gender":12}


api.add_resource(ProductImageProcess, "/validate-image")
api.add_resource(GenderImageProcess, "/validate-kyc")

if __name__ == "__main__":
    app.run(debug=True)