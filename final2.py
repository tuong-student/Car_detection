import netrc
from re import X
import tkinter as tk
from tracemalloc import start
from turtle import color, heading
from unicodedata import name
import PIL
from tkinter.constants import NS, NW, Y
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
from matplotlib.pyplot import fill
from pyparsing import col
import numpy as np
import time 
import pafy
window = tk.Tk()
window.title("Car detection - group 02")

def openFile():
    filepath = askopenfilename(filetypes=[("Text File", "*.mp4"),("All Files", "*.*")])

    if not filepath:
        return
    txt_edit.delete("1.0", tk.END)
    txt_edit.insert(tk.END, filepath)

bigFrame = tk.Frame(window, width=200) #contain frame0 and frame1
frame0 = tk.Frame(bigFrame, width=200)
main_lable = tk.Label(frame0, text="Group 02 \n Nguyễn Thanh Tường-19110066 \n Phạm Hoàng Minh Mẫn-19110066 \n Huỳnh Công Đạt-19110114 \n Nguyễn Khang Duy-19110066", height=5)
main_lable.pack(ipadx=150)
frame1 = tk.Frame(bigFrame, width=200)
txt_edit = tk.Text(frame1, padx=5, pady=5, height=1)
txt_edit.pack(fill=tk.X, padx=5)

#load yolo
net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#dnn is a deep neron network

classes = ["car"]
#Select the object want to count

with open("coco.names", "r") as f:
    #load class name list from coco
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()   # get layer_names from yolov3 (list)
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]     # get tracked object layer from layer_names
colors = np.random.uniform(0, 255, size = (len(classes), 3))    # sign color for layer 
frame_id = 0

def getLink():
    name = txt_edit.get(1.0, tk.END)
    name.strip()
    name2 = name[0:int(len(name)) - 1]
    return name2

cap = cv2.VideoCapture()
#Is Used to URL

starting_time = 0
def showVideo():
    cap.open(getLink())
    global starting_time
    starting_time = time.time()
    videoStream()

def videoStream():
    front = cv2.FONT_HERSHEY_PLAIN
    global starting_time
    _, frame = cap.read()
    #frame is a vector array is capture base on default fps
    #cap.read() loop frame until the end or error, return bool if the frame is read correctly

    global frame_id     # used to know which frame we are
    frame_id += 1
    height, width, channels = frame.shape
    #channels is number of color channel
    # cv2.imshow("result video", frame)
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
    #convert image to blob image 
    #frame, out_size = (416x416), scaleFactor = 0.00392, mean = (0,0,0)
    #scaleFactor: divide image ratition
    #mean = mean: the avg value of RGB channel
    #because we don't want to change the color so value is (0,0,0)
    #SwapRB = True will swap R and B channel to conver BGR image into RGB image  
    #Blod was used to output atr from image and change their size, Yolo accept 3 size:
    # 320x320: fast, less acc
    # 609x609: slower, more acc
    # 416x416: medium - standard
    # 608x608: slow, more acc

    net.setInput(blob)
    outs = net.forward(output_layers)
    # net.forward give a list of objects are detected with height, width, name, x y coordiante....

    class_ids = []
    confidences = []
    #Confidences store confident point of class_id if confident point > 0.2 the object will be detect
    #Then we find the center point 
    #Only when confidence > confidence threshold, class_id will be added
    boxes =[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            #argmax return the max value of the array
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int (detection[0] * width)
                center_y = int (detection[1] * height)
                w = int (detection[2] * width)
                h = int (detection[3] * height)

                #Cordinate
                x = int (center_x - w/2)
                y = int (center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    #To avoid many box for one object, we use Non Maximum Suppresion (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    #Boxes, conf, con_threshold, nms_threshold
    #box: contain the cordinate of rectangle cover the object
    #confidences: confident level 0 -> 1
    #conf_threshold = 0.8: if the value of the object less than 0.8, model will pass it
    #nms_threshold = 0.3: if the object has 2 box at the same time, and the area is over 0.3, 1 box will be deleted
    d = 0
    for i in range(len (boxes)):
        if i in indexes:
            d = d + 1
            x, y, w, h = boxes[i]
            #output cordinate from box
            #sign color for box
            lable = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, lable + " " + str(round(confidence, 2)), (x, y + 30), front, 2, color, 3)
            #cv2.putText: sign name (persion, car...) for the object

    
    number.delete("1.0", tk.END)
    number.insert(tk.END, d)
    #Compute FPS 
    elapsed_time = time.time() - starting_time   # time have runed
    fps = (frame_id / elapsed_time)
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (170,50), front, 2, (0,0,0), 2)     # show FPS (2 number after .)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    la4.imgtk = imgtk
    la4.configure(image=imgtk)
    la4.after(10, videoStream)

window.rowconfigure([0,1,2], minsize=50)
window.columnconfigure([0, 1, 2], minsize=50)
la1 = tk.Label(frame1, text="Muc tuy chon: ", width=10)
button1 = tk.Button(frame1, width=10, height=1, text="File video", pady=5, padx=5, command=openFile)
button2 = tk.Button(frame1, width=10, height=1, text="Show video", pady=5, padx=5, command=showVideo)
la6 = tk.Label(frame1, text="Number ", padx=6, pady=6, width=10)
number = tk.Text(frame1, padx=5, pady=5, width=10, height=1)

frame2 = tk.Frame(window, width=350)

la2 = tk.Label(frame2, text="Result video", padx=5, width=10, height=1)
la4 = tk.Label(frame2, width=540, height=450, pady=5)
la4.grid()
la2.grid()
bigFrame.grid(row=0, column=0, sticky=NS)
frame2.grid(row=0, column=1, sticky=NS)
frame0.grid(row=0, column=0, sticky=NS)
frame1.grid(row=1, column=0, sticky=NS)
la1.pack(fill = tk.X)
button1.pack()
button2.pack()
la6.pack(fill=tk.X)
number.pack()
window.mainloop()