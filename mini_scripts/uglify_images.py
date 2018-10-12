import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 1: View without saving, 2: Save files without showing
review=1;

# dir_glob ='../src/images/OMR_Files'+'/*/*/*/*.jpg'
dir_glob ='hist_inputs'+'/*.jpg'
u_width=1000

def waitQ():
    while(0xFF & cv2.waitKey(1) != ord('q')):pass
    cv2.destroyAllWindows()

show_count=0
def show(img,title="",wait=True):     
    global show_count
    if(title==""):
        show_count+=1
        title="Image "+str(show_count)
        
    cv2.imshow(title,img)
    if(wait):
    	waitQ()

def showOrSave(filepath,img,title="",wait=True):
	if(review):
		show(img,title,wait)
	elif(wait):
		filename=filepath[filepath.rindex("/")+1:]
		cv2.imwrite("hist_outputs/"+filename,img)

def stitch(img1,img2):
	if(img1.shape!=img2.shape):
		print("Can't stitch different sized images")
		return None
	return np.concatenate((img1,img2),axis=1)

allOMRs= glob.iglob(dir_glob)
for filepath in allOMRs:
	print (filepath)
	img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	h,w=img.shape
	img = cv2.resize(img,(u_width,int(h*u_width/w)))
	orig=img.copy()
	
	showOrSave(filepath,img,wait=False)
	# for i in range(10,360,10):
		# img = cv2.rotate(i);	
		# showOrSave(filepath,img)

		
