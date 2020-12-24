import cv2

cap = cv2.VideoCapture('./dataset/clip1.mp4')
succ, frame_bgr = cap.read()
frm = 1
while succ:
    cv2.imwrite("./dataset/frames/clip1/{}.jpg".format(frm), frame_bgr)
    succ, frame_bgr = cap.read()
    frm += 1
    print('write frame', frm)



