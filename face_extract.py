# CACH DUNG LENH
# python face_extract.py --input_dir videos/ --embeddings output/embeddings.pickle --embedding-model openface_nn4.small2.v1.t7

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", type=str, required=True,
                help="path to input video")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-s", "--skip", type=int, default=1,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []
# initialize the total number of faces processed
total = 0

# Doc file video input
for filename in os.listdir(args["input_dir"]):
    input = args["input_dir"] + filename
    # lay label
    name = filename.split('.')[0]
    # bat video, tach frame va xu li tung frame
    vs = cv2.VideoCapture(input)
    read = 0
    saved = 0

    # Lap qua cac frame cua video
    while True:

        (grabbed, frame) = vs.read()
        # Neu khong doc duoc frame thi thoat
        if not grabbed:
            break

        read += 1
        if read % args["skip"] != 0:
            continue

        # Chuyen tu frame thanh blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Phat hien cac khuon mat trong frame
        detector.setInput(blob)
        detections = detector.forward()

        # Neu tim thay it nhat 1 khuon mat
        if len(detections) > 0:
            # Tim khuon mat to nhat trong anh
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Neu muc do nhan dien > threshold
            if confidence > args["confidence"]:
                # Tach khuon mat va ghi ra file
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    vs.release()
    cv2.destroyAllWindows()

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()