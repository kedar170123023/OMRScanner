import cv2
import glob
import imutils
import unittest
import numpy as np
from random import random
from math import sin,cos,pi as PI
from matplotlib import pyplot as plt
# 1: View without saving, 2: Save files without showing
review=1;

# dir_glob ='../src/images/OMR_Files'+'/*/*/*/*.jpg'
dir_glob ='hist_inputs'+'/*.jpg'
bg_glob = 'omrbgs/omrbg*.jpg'
u_width=600

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

def showOrSave(filepath,orig,title="",wait=True,forced=False):
	global review
	h,w=orig.shape[:2]
	u_height = int(h*u_width/w)
	img = cv2.resize(orig,(u_width,u_height))
	if(review):
		show(img,title,wait)
	elif(save or wait):
		filename=filepath[filepath.rindex("/")+1:]
		cv2.imwrite("hist_outputs/"+filename,img)

def rotateLine(p,q,a):
	a *= PI/180
	x0,y0 = q
	x1,y1 = p
	p[0] = int(((x1 - x0) * cos(a)) - ((y1 - y0) * sin(a)) + x0);
	p[1] = int(((x1 - x0) * sin(a)) + ((y1 - y0) * cos(a)) + y0);
	return p

def drawPoly(img, pts,color=(255,255,255), thickness=10):
	l = len(pts)
	for i in range(0,l+1):
		cv2.line(img,tuple(pts[(i-1)%l]),tuple(pts[i%l]),color=color, thickness=thickness)

def zeroPad(img, padDiv = 10):	
	h, w = img.shape[:2]
	bg = np.zeros((int((1+2/padDiv)*h),int((1+2/padDiv)*w)), np.uint8)
	x,y,wi,hi = w//padDiv,h//padDiv, w//2, h//2
	c = [x+wi,y+hi]
	pts=[[-wi,-hi],[-wi,hi],[wi,hi],[wi,-hi]]
	bg[y:(y+h) , x:(x+w)] = img;
	return bg,c,pts
    
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    minWidth = max(int(widthA), int(widthB))
    # minWidth = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # maxHeight = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [minWidth - 1, 0],
        [minWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (minWidth, maxHeight))

    # return the warped image
    return warped
    
class testImageWarps(unittest.TestCase):

	def setUp(self):
		self.allIMGs=[]
		allOMRs= glob.iglob(dir_glob)
		allBGs= glob.iglob(bg_glob)
		bgs=[]
		for bgpath in allBGs:
			bgs.append(
				cv2.normalize(
					cv2.imread(bgpath, cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
					)
				)

		bglen = len(bgs)

		for i,filepath in enumerate(allOMRs):
			img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
			h,w=img.shape[:2]
			u_height = int(h*u_width/w)
			img = cv2.resize(img,(u_width,u_height))
			img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			
			scale = 2.0 + random()
			bg = cv2.resize(bgs[i%bglen],(int(scale*u_width),int(scale*u_height)))
			# M1: np.zeroes as used elsewhere in this code
			# M2: img = cv2.copyMakeBorder(img,u_height//2,u_height//2,u_width//2,u_width//2,cv2.BORDER_CONSTANT, value=(0,0,0))
			# M3: Can also warp the mask, but its better to keep bg aligned with edges of sheets
			# M4: imutils! But will loose track of pts (or will we),https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/ 
			# https://stackoverflow.com/questions/32266605/warp-perspective-and-stitch-overlap-images-c#_=__name__
			x,y,w,h = u_width//2,u_height//2,u_width, u_height
			# use list than tuple to support list assignment
			pts=[[x,y],[x,y+h],[x+w,y+h],[x+w,y]]
			bg[y:(y+h) , x:(x+w)] = img;
			
			img = bg
			# img = cv2.resize(bg,(u_width,u_height))
			
			self.allIMGs.append((img,pts, filepath))
			break

	def testPerspective(self):
		#  perspective on whole image (including bg)
		for baseimg, pts, filepath in self.allIMGs:
			h,w=baseimg.shape[:2]
			img=baseimg.copy()						


# TODO: Refactor and resolve from here: 
			# Padding for persp
			img, center, pts1 = zeroPad(img,padDiv=5)
			h,w=img.shape[:2]

			thetaBase,thetaDist,thetaWide = 25,15,5
			# rotate image about midpoint
			# img = imutils.rotate_bound(img, thetaBase)
			hi,wi = img.shape[:2]
			newcenter = [wi//2,hi//2]
			pt=newcenter
			img = cv2.rectangle(img,tuple(pt),(pt[0]+30,pt[1]+50),(10,10,10),10)  
			
			# rotate and shift pts1 to new center 
			for p in pts1:
				p[0]+=newcenter[0]
				p[1]+=newcenter[1]
				# p = rotateLine(p,newcenter,thetaBase)
			
			drawPoly(img,pts1,color=(205,0,0))
			
			# Create inverse warp rectangle
			pts2 = pts1.copy()
			pts2[3] = rotateLine(pts2[3],pts2[0],-thetaDist)
			pts2[0] = rotateLine(pts2[0],pts2[3],thetaDist)
			pts2[0] = rotateLine(pts2[0],pts2[1],thetaWide)
			pts2[3] = rotateLine(pts2[3],pts2[2],thetaWide)

			# Done - draw above as dotted lines on image.
			# cv2.polylines(img,[np.array(pts2, np.int32)],isClosed=True,color=(255,255,255), thickness=10)
			# % is positive in python
			drawPoly(img,pts2,color=(205,0,0))
			showOrSave(filepath,img)

			# Apply perspective
			# M = cv2.getPerspectiveTransform(np.float32(pts1),np.float32(pts2))
			# img = cv2.warpPerspective(img,M,(wi,hi))
			img = four_point_transform(img,np.array(pts2))

			showOrSave(filepath,img)
	
	# def testRotation(self):
	# 	for baseimg, pts, filepath in self.allIMGs:
	# 		h,w=baseimg.shape[:2]
	# 		# self.assertEqual(w,u_width, 'width not resized properly!')
	# 		img=baseimg.copy()
	# 		for i in range(-40,41,10):
	# 			# first 2 the coordinate limits.
	# 			M = cv2.getRotationMatrix2D((h//2,w//2),i,scale=1)
	# 			# print("M",M.shape, M)
	#			# third arg is the output image size
	# 			showOrSave(filepath,cv2.warpAffine(img,M,(w,h)))

	# def testTranslate(self):
	# 	for baseimg, pts, filepath in self.allIMGs:
	# 		h,w=baseimg.shape[:2]
	# 		img=baseimg.copy()
	# 		for i in range(-w//5,w//5,w//5):
	# 			for j in range(-h//5,h//5,h//5):
	# 				showOrSave(filepath,cv2.warpAffine(img,np.float32([[1,0,i],[0,1,j]]),(w,h)))


# if run as script and not imported as pkg
# if __name__ == '__main__' :
	# unittest.main()
suite = unittest.TestLoader().loadTestsFromTestCase(testImageWarps)
# 2 is max verbosity offered
unittest.TextTestRunner(verbosity=2).run(suite) 