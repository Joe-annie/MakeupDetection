import os
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.6/site-packages/cv2/data/Mouth.xml')


img_path0 = './makeup_with_labels/no_makeup'
path_save0 = './makeup_with_labels/no_makeup_face'

for idx,fn in enumerate(os.listdir(img_path0)):
    if idx > 0: break
    if fn.endswith('.jpg'):
        #read img
        fd = os.path.join(img_path0,fn)
        img = cv2.imread(fd)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        if len(faces) < 1: # no face detect
            roi_gray = gray
            roi_color = img
        else:
            for (x, y, w, h) in faces: # detect face
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
        img_face = roi_color
        #存在本地
        cv2.imwrite(fn + 'face' + '.jpg', img_face)


        # Eye
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 3)
        for idx,(ex, ey, ew, eh) in enumerate(eyes):
            # img_eye = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # crop roi img
            img_eye = roi_color[ey:ey + eh,ex:ex + ew]
            #存在本地
            cv2.imwrite(fn + 'eye_'+ str(idx+1)+ '.jpg', img_eye)



        # Mouth & Nose
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.2, 3)
        for idx,(mx, my, mw, mh) in enumerate(mouth):
            #img_mouth = cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            # crop roi img
            img_mouth = roi_color[my:my + mh,mx:mx + mw]
            #存在本地
            cv2.imwrite(fn + 'mouth_'+ str(idx+1)+ '.jpg', img_mouth)



        # cv2.namedWindow('Person Detected!')
        # cv2.imshow('Person Detected!', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

