import numpy as np
import cv2
import dlib

Image_size = 64

def face_catch(img,filename_f):
    detector = dlib.get_frontal_face_detector()
    
    face_rects,score,idx = detector.run(img,0)
    for i,d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        new_img_name =  'test_image/other_test/face_' + filename_f + '.jpg'
        print(new_img_name)
        img_save = img[y1:y2,x1:x2]
        if(i == -1):
            pass
        else:
            cv2.imwrite(new_img_name,img_save)


def face_catch_api(filename):
    print(filename)
    filename_f = filename.split('\\')[-1].split('.')[0]
    img = cv2.imread(filename)
    face_catch(img,filename_f)


if __name__ == '__main__':
    filename = 'test_image/other_test/022.jpg'
    filename_f = filename.split('/')[-1].split('.')[0]
    img = cv2.imread(filename)
    face_catch(filename)