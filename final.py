import os
import cv2
import sys
import time
import torch
import argparse
import imutils
import numpy as np

from fastai import *
from fastai.vision import *
from torchvision import transforms
from imutils.video import FPS
from imutils.video import VideoStream

#from key_so import switch_program
from keypresses import PressKey, ReleaseKey

from infer_unet_mine import predict

UP = 0xC8
LEFT = 0xCB
RIGHT = 0xCD
DOWN = 0xD0
LCONTROL = 0x1D
SPACE = 0x39


def send_key_simulation(key):
    if key=='a':
        PressKey(UP)
        time.sleep(0.05)
        ReleaseKey(UP)
    elif key=='b':
        PressKey(LEFT)
        time.sleep(0.05)
        ReleaseKey(LEFT)
    elif key=='c':
        PressKey(SPACE)
        time.sleep(0.05)
        ReleaseKey(SPACE)
    elif key=='d':
        PressKey(RIGHT)
        time.sleep(0.05)
        ReleaseKey(RIGHT)
    elif key=='e':
        time.sleep(0.05)
    return
        

def extractSkin(image):
  # Taking a copy of the image
  img =  image.copy()
  # Converting from BGR Colours Space to HSV
  img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  
  # Defining HSV Threadholds
  lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
  upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
  
  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
  
  # Cleaning up mask using Gaussian Filter
  #skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  
  # Extracting skin from the threshold mask
  skin  =  cv2.bitwise_and(img,img,mask=skinMask)
  
  # Return the Skin image
  return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)
'''
def remove_background(background,frame):
	#frame= cv2.GaussianBlur(frame, (5, 5), 0)
	lower_threshold = np.array([10,10,10], dtype=np.uint8)
	upper_threshold = np.array([255, 255, 255], dtype=np.uint8)
	difference = cv2.absdiff(background, frame)
	#cv2.imshow('difference',difference)
	skinMask = cv2.inRange(difference,lower_threshold,upper_threshold)
	#cv2.imshow('skinmask',skinMask)
	skin  =  cv2.bitwise_and(frame,frame,mask=skinMask)
	# Return the Skin image
	return skin


	#ret, thresh = cv2.threshold(difference, 5, 255, cv2.THRESH_BINARY)
	#cv2.imshow('thresh',thresh)
	#(_, cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#skin = max(cnts, key=cv2.contourArea)
	
	#ret, thresh1 = cv2.threshold(thresh, 10, 255, cv2.THRESH_BINARY_INV)
	#cv2.imshow('thresh diff',thresh)
	#ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV)
	#cv2.imshow('thresh inv',thresh)
	#skin= cv2.bitwise_and(frame, thresh)
	return thresh
'''

def remove_background(frame):
    output, mask = predict(net, frame, normalize, opt.gpu)
    frame = cv2.resize(frame, (240,240), interpolation = cv2.INTER_AREA)
    frame_new = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow("mask",mask)
    return frame_new

def remove_background_manual(background,frame):
	frame1=frame
	for i in range(len(frame)):
		for j in range (len(frame[0])):
			sum_frame=sum(frame[i][j])
			sum_background=sum(background[i][j])
			if abs(sum_background - sum_frame)<10:
				frame1[i][j]=[0,0,0]
	return frame1



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='network.pth', help='model path')
parser.add_argument('--outf', type=str, default='./test_results', help='output folder')
#parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--gpu', action='store_true', help="use feature transform")
opt = parser.parse_args()

if not opt.gpu:
	net = torch.load(opt.model, map_location=torch.device('cpu'))
else:
	net = torch.load(opt.model).cuda()
	print("Using GPU for segmentation")

normalize = transforms.ToTensor()
net.eval()

if not os.path.isdir(opt.outf):
	os.mkdir(opt.outf)

print('Hello, I hope you are having a good day.')
print("[INFO] loading model...")

path=''
#learn = load_learner(path =r"C:\Users\Sylar\Desktop\Product_11", fname= “export.pkl”)
learn=load_learner(path=path)

print("[INFO] model loaded...")

ans='' 
alphabet_array=[]

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
print('Press enter to start')
input()
'''
print('Press enter to capture background')
input()
aweight=0.5
for i in range(30):

	temp=vs.read()
	top, right,bottom, left = 120, 350, 360, 590 #(590,10  ,  350,225) 120 360
	temp= temp[top:bottom, right:left]
	#temp = cv2.GaussianBlur(temp, (7, 7), 0)
	if i==0:
		background=temp
		background=background.astype("float")

	else:
		cv2.accumulateWeighted(temp,background, aweight)
#background= cv2.GaussianBlur(background, (5, 5), 0)
background=background.astype("uint8")
'''
#cv2.imshow('background',background)
counter=0
recapture=1

#print("Switching to game screen...")
#switch_program()

while True:
	recapture+=1

	'''
	if recapture%300==0:
		print('background recapture. press enter')
		input()
		background=vs.read()
		background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

		top, right,bottom, left = 120, 350, 360, 590 #(590,10  ,  350,225) 120 360
		background= background[top:bottom, right:left]
	'''
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()

	top, right,bottom, left = 120, 350, 360, 590#(590,10  ,  350,225) 120 360
	cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
		
	roi = frame[top:bottom, right:left]
	#roi = cv2.GaussianBlur(roi, (7, 7), 0)
	#frame_skinned=extractSkin(roi)
    
	#frame_skinned=remove_background(background,roi)
	frame_skinned=remove_background(roi)
    
	#frame_skinned= frame_skinned[top:bottom, right:left]
	#frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,50)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2
	
	#cv2.imwrite('temp.jpg',frame_skinned)
	temp = Image(pil2tensor(frame_skinned, dtype=np.float32).div_(255))
	#temp = Image(torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2,0,1)).float()/255)
	#temp=open_image('temp.jpg')
	values=np.array(learn.predict(temp))[2]
	temp=np.argmax(values)
	confidence=values[temp]
	alphabet=temp+97
	alphabet_char=chr(alphabet)
	
	char_flag=False
	
	if len(alphabet_array)>8:
		del alphabet_array[0]
		alphabet_array.append(alphabet_char)
	else:
		alphabet_array.append(alphabet_char)

	sorted_alphabet_array=sorted(alphabet_array)
	most_freq_char=sorted_alphabet_array[-1]
	most_freq_char_count=sorted_alphabet_array.count(most_freq_char)

	if most_freq_char_count>4 and len(alphabet_array)>7: 
		ans=most_freq_char
		alphabet_array=[]

	alphabet=chr(alphabet) +'  '+str(confidence)
	alphabet1=ans
	
	send_key_simulation(ans)

	cv2.putText(frame,alphabet1, 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    lineType)
	
	#cv2.imshow("Frame", frame)

	cv2.putText(frame_skinned,alphabet, 
	    bottomLeftCornerOfText, 
	    font, 
	    fontScale,
	    fontColor,
	    lineType)

	cv2.imshow("frame_skinned",frame_skinned)
	#cv2.imshow('Frame1',frame)
	counter+=1
		
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		 
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()
		sys.exit(0)
 
	# update the FPS counter
	fps.update()

	fps.stop()
