import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import unittest
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


class testImageWarps(unittest.TestCase):
	def setUp(self):
		self.allIMGs=[]
		allOMRs= glob.iglob(dir_glob)
		for filepath in allOMRs:
			img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
			h,w=img.shape
			img = cv2.resize(img,(u_width,int(h*u_width/w)))
			self.allIMGs.append(img)

	def testRotation(self):
		for img in self.allIMGs:
			h,w=img.shape
			orig=img.copy()
			self.assertEqual(w,u_width, 'width not resized properly!')

# if run as script and not imported as pkg
# if __name__ == '__main__' :
	# unittest.main()
suite = unittest.TestLoader().loadTestsFromTestCase(testImageWarps)
unittest.TextTestRunner(verbosity=3).run(suite)