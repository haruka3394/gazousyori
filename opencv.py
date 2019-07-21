import cv2
import numpy as np
import matplotlib.pyplot as plt

y=[]

capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    avr = np.average(gray)
    print(avr)
    y.append(avr)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print(len(y))
x=np.linspace(1,100,len(y))
capture.release()
cv2.destroyAllWindows()
plt.plot(x, y, label="test")
plt.show()
