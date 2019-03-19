import cv2
import json
import numpy as np
from constants import *

def resize_util(img, u_width, u_height=None):
    if u_height == None:
        h,w=img.shape[:2]
        u_height = int(h*u_width/w)        
    return cv2.resize(img,(u_width,u_height))
### Image Template Part ###
template = cv2.imread('images/FinalCircle_hd.png',cv2.IMREAD_GRAYSCALE) #,cv2.CV_8UC1/IMREAD_COLOR/UNCHANGED 
template = resize_util(template, int(template.shape[1]/templ_scale_down))
template = cv2.GaussianBlur(template, (5, 5), 0)
template = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# template_eroded_sub = template-cv2.erode(template,None)
template_eroded_sub = template - cv2.erode(template, kernel=np.ones((5,5)),iterations=5)
lontemplateinv = cv2.imread('images/lon-inv-resized.png',cv2.IMREAD_GRAYSCALE)
# lontemplateinv = imutils.rotate_bound(lontemplateinv,angle=180) 
# lontemplateinv = imutils.resize(lontemplateinv,height=int(lontemplateinv.shape[1]*0.75))
# cv2.imwrite('images/lontemplate-inv-resized.jpg',lontemplateinv)

### Coordinates Part ###
class Pt():
    """Container for a Point Box on the OMR"""
    def __init__(self, x, y,val):
        self.x=x
        self.y=y
        self.val=val
    # overloaded
    def __init__(self, pt,val):
        self.x=pt[0]
        self.y=pt[1]
        self.val=val

class Q():
    """
    Container for a Question on the OMR
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """
    def __init__(self,qNo,qType, pts,ans=None):
        self.qNo = qNo
        self.qType = qType
        self.pts = pts
        self.ans = ans

class Qblock():
    def __init__(self,orig, dims, Qs):
        self.orig=orig
        self.dims=dims
        self.Qs=Qs

class Template():
    def __init__(self):
        self.Qblocks = []
        self.boxDims = [-1, -1]
        self.dims = [-1,-1]

    def setDims(dims):
        self.dims = dims

    def setBoxDims(dims):
        self.boxDims = dims

    def addQblock(rect):
        self.Qblocks.append(Qblock(rect.orig, calcQBlockDims(rect), maketemplate(rect)))

def genRect(orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - values of each alternative for a question

    Output:
    Returns set of coordinates of a rectangular grid of points
    
        1 2 3 4
        1 2 3 4
        1 2 3 4

        (q1, q2, q3)

        00
        11
        22
        33
        44
        (q1d1,q1d2)

    """
    Qs=[]
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    o=orig[:] # copy list
    for qNo in qNos:
        pt = o[:] #copy pt
        pts=[]
        for v in vals:
            pts.append(Pt(pt,v))
            pt[i0] += gaps[i0]
        o[i1] += gaps[i1]
        Qs.append( Q(qNo,qType, pts))
    return Qs

def genGrid(orig, qNos, bigGaps, gaps, vals, qType, orient='V'):
    """
    Input:
    orig- start point
    qNos - an array of qNos tuples(see below) that align with dimension of the big grid (gridDims extracted from here)
    bigGaps - (bigGapX,bigGapY) are the gaps between blocks
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - a 1D array of values of each alternative for a question
    orient - The way of arranging the vals (vertical or horizontal)

    Output:
    Returns an array of Q objects (having their points) arranged in a rectangular grid

                                00    00    00    00
   Q1   1 2 3 4    1 2 3 4      11    11    11    11
   Q2   1 2 3 4    1 2 3 4      22    22    22    22         1234567
   Q3   1 2 3 4    1 2 3 4      33    33    33    33         1234567
                                44    44    44    44
                            ,   55    55    55    55    ,    1234567                       and many more possibilities!
   Q7   1 2 3 4    1 2 3 4      66    66    66    66         1234567
   Q8   1 2 3 4    1 2 3 4      77    77    77    77
   Q9   1 2 3 4    1 2 3 4      88    88    88    88
                                99    99    99    99

    MCQ type (orient='H')-
        [
            [(q1,q2,q3),(q4,q5,q6)]
            [(q7,q8,q9),(q10,q11,q12)]
        ]

    INT type (orient='V')-
        [
            [(q1d1,q1d2),(q2d1,q2d2),(q3d1,q3d2),(q4d1,q4d2)]
        ]
    
    ROLL type-
        [
            [(roll1,roll2,roll3,...,roll10)]
        ]

    """
    npqNos=np.array(qNos)
    if(len(npqNos.shape)!=3 or npqNos.size==0): # product of shape is zero
        print("genGrid: Invalid qNos array given", npqNos)
        return []

    # ^ should also validate no overlap of rect points somehow?!
    gridHeight, gridWidth, numDigs = npqNos.shape
    numVals = len(vals)
    # print(orig, numDigs,numVals, gridWidth,gridHeight, npqNos)

    Qs=[]
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    hGap, vGap = bigGaps[i1], bigGaps[i0]
    if(orient=='H'):
        hGap += (numVals-1)*gaps[i1]
        vGap += (numDigs-1)*gaps[i0]
    else:
        hGap += (numDigs-1)*gaps[i1]
        vGap += (numVals-1)*gaps[i0]
    qStart=orig[:]
    for row in npqNos:
        qStart[i1] = orig[i1]
        for qTuple in row:
            Qs += genRect(qStart,qTuple,gaps,vals,qType,orient)
            qStart[i1] += hGap
        qStart[i0] += vGap

    return Qs

# The utility for GUI            
def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)


def calcQBlockDims(rect):
    # 
    rect.
    return (dimsX,dimsY)


def scalePts(pts,facX,facY):
    for pt in pts:
        pt = (pt[0]*facX,pt[1]*facY)

def read_template(filename):    
    with open(filename, "r") as f:
        return json.load(f)


# Config for Manual fit - 
templJSON={
'J' : read_template("J_template.json"),
'H' : read_template("H_template.json")
}

qtype_data = {
'QTYPE_MED':{
'vals' : ['E','H'],
'orient':'V'
},
'QTYPE_ROLL':{
'vals':range(10),
'orient':'V'
},
'QTYPE_INT':{
'vals':range(10),
'orient':'V'
},
'QTYPE_MCQ':{
'vals' : ['A','B','C','D'],
'orient':'H'
},
}

TEMPLATES={'J': Template(),'H': Template()}
def maketemplate(rect):
    # keyword arg unpacking followed by named args
    return genGrid(**rect,**qtype_data[rect['qType']])


for squad in ['J','H']:
    for k, rect in templJSON[squad].items():
        if(k=="Dimensions"):
            TEMPLATES[squad].setDims(rect)
            continue
        if(k=="boxDimensions"):
            TEMPLATES[squad].setBoxDims(rect)
            continue
        # Internal adjustment: scale fit
        scalePts([rect['orig'],rect['bigGaps'],rect['gaps']],omr_templ_scale[0],omr_templ_scale[1])
        # Add QBlock to array of grids
        TEMPLATES[squad].addQblock(rect)

    if(TEMPLATES[squad].dims != [-1, -1]):
        print("Invalid JSON! No reference dimensions given")
