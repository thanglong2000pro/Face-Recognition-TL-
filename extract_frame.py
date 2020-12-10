# CACH DUNG LENH
# python extract_frame.py --input_dir videos/ --output_dir dataset/

import argparse
import cv2
import os

# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output_dir", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-s", "--skip", type=int, default=1,
                help="# of frames to skip before applying face detection")
ap.add_argument("-n", "--num-to-train", type=int, default=80,
                help="# num of frames to train")
args = vars(ap.parse_args())

# Doc file video input
for filename in os.listdir(args["input_dir"]):
    input = args["input_dir"] + filename
    vs = cv2.VideoCapture(input)
    num_frame = vs.get(cv2.CAP_PROP_FRAME_COUNT)

    no_frame = 0
    read = 0
    saved = 0
    frameRate = int(num_frame / args['num_to_train'])

    # Lap qua cac frame cua video
    while True:

        vs.set(cv2.CAP_PROP_POS_FRAMES, no_frame)
        (grabbed, frame) = vs.read()
        # Neu khong doc duoc frame thi thoat
        if not grabbed:
            break

        read += 1
        if read % args["skip"] != 0:
            continue

        # write the frame to disk
        output = args["output_dir"] + filename
        # tach duoi mp4
        output = output.split('.')
        # lay label
        name = (output[0].split('/'))[1]
        # duong dan luu file
        p = os.path.sep.join([output[0],
                                  input.split('/')[1] + "{}.png".format(saved)])
        cv2.imwrite(p, frame)
        saved += 1
        no_frame += frameRate
        print("[INFO] saved {} to disk".format(p))

vs.release()
cv2.destroyAllWindows()
