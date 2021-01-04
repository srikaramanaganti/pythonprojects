import cv2
font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cap = cv2.VideoCapture("Free Stock Footage People Walking Talking.mp4")
harcascade=cv2.CascadeClassifier("../haarcascade_fullbody.xml")
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    full_body = harcascade.detectMultiScale(gray,1.3,2)
    for (x,y,w,h) in full_body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow('screen',frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
