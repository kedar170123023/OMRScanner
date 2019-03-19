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

class QBlock():
    def __init__(self, dims, orig, Qs):
        # dims = (width, height)
        self.dims = dims
        self.orig = orig
        self.Qs = Qs
        # will be set when using
        self.shift = 0

class Template():
    def __init__(self):
        self.QBlocks = []
        self.boxDims = [-1, -1]
        self.dims = [-1,-1]

    def setDims(self,dims):
        self.dims = dims

    def setBoxDims(self,dims):
        self.boxDims = dims

    # Expects boxDims to be set already
    def addQBlocks(self, rect):
        # keyword arg unpacking followed by named args
        self.QBlocks += genGrid(self.boxDims, **rect,**qtype_data[rect['qType']])
        # self.QBlocks.append(QBlock(rect.orig, calcQBlockDims(rect), maketemplate(rect)))

def genQBlock(QBlockDims, orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - values of each alternative for a question

    Output:
    // Returns set of coordinates of a rectangular grid of points
    Returns a QBlock containing array of Qs and some metadata
    
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
    H, V = (0,1) if(orient=='H') else (1,0)
    o = orig[:] # copy list  <-- Not for multi dimensions tho!
    for qNo in qNos:
        pt = o[:] #copy pt
        pts=[]
        for v in vals:
            pts.append(Pt(pt,v))
            pt[H] += gaps[H]
        o[V] += gaps[V]
        Qs.append( Q(qNo,qType, pts))
    
    return QBlock(QBlockDims, orig, Qs)

def genGrid(boxDims, orig, qNos, bigGaps, gaps, vals, qType, orient='V'):
    """
    Input:
    boxDims - dimesions of single QBox
    orig- start point
    qNos - an array of qNos tuples(see below) that align with dimension of the big grid (gridDims extracted from here)
    bigGaps - (bigGapX,bigGapY) are the gaps between blocks
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - a 1D array of values of each alternative for a question
    orient - The way of arranging the vals (vertical or horizontal)

    Output:
    // Returns an array of Q objects (having their points) arranged in a rectangular grid
    Returns grid of QBlock objects

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
    gridData=np.array(qNos)
    if(len(gridData.shape)!=3 or gridData.size==0): # product of shape is zero
        print("genGrid: Invalid qNos array given", gridData)
        return []

    # ^ should also validate no overlap of rect points somehow?!

    """
    orient = 'H'
    numVals = 4
    [
    [["q1", "q2", "q3", "q4"], ["q5", "q6", "q7", "q8"]],
    [["q9", "q10", "q11", "q12"], ["q13", "q14", "q15", "q16"]]
    ]

    q1          q9
    q2          q10
    q3          q11 
    q4          q12

    q5          q13 
    q6          q14
    q7          q15
    q8          q16
    """

    gridRows, gridCols = gridData.shape[:2]
    numQsMax = max([max([len(qb) for qb in row]) for row in gridData])
    numVals = len(vals)

    QBlocks=[]
    # H and V are named with respect to orient == 'H', reverse their meaning when orient = 'V'
    H, V = (0,1) if(orient=='H') else (1,0)
    numDims = [numQsMax, numVals]
    print(orig, numDims, gridRows,gridCols, gridData)
    # orient is also the direction of making QBlocks
    # Simple is powerful-
    origGap = [
        bigGaps[V] + (numDims[V]-1)*gaps[V],
        bigGaps[H] + (numDims[H]-1)*gaps[H]
    ]
    
    qStart = orig[:] #copy list
    for row in gridData:        
        qStart[V] = orig[V]
        for qTuple in row:
            QBlockDims = [
                # width x height in pixels
                gaps[H] * (numDims[V]-1) + boxDims[H],
                gaps[V] * (numDims[H]-1) + boxDims[V]
            ]
            # Qs += genQBlock(qStart,qTuple,gaps,vals,qType,orient)
            QBlocks.append(genQBlock(QBlockDims, qStart,qTuple,gaps,vals,qType,orient))
            # TODO : Verify [0]
            qStart[V] += origGap[0]
        qStart[H] += origGap[1]

    return QBlocks

# The utility for GUI            
def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)

def scalePts(pts,facX,facY):
    for pt in pts:
        pt = (pt[0]*facX,pt[1]*facY)

def read_template(filename):    
    with open(filename, "r") as f:
        return json.load(f)



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

templJSON={
'J' : read_template("J_template.json"),
'H' : read_template("H_template.json")
}
TEMPLATES={'J': Template(),'H': Template()}


for squad in ['J','H']:
    TEMPLATES[squad].setDims(templJSON[squad]["Dimensions"])
    TEMPLATES[squad].setBoxDims(templJSON[squad]["boxDimensions"])
    for k, rect in templJSON[squad].items():
        if(k=="Dimensions" or k=="boxDimensions"):
            continue
        # Internal adjustment: scale fit
        scalePts([rect['orig'],rect['bigGaps'],rect['gaps']],omr_templ_scale[0],omr_templ_scale[1])
        # Add QBlock to array of grids
        TEMPLATES[squad].addQBlocks(rect)

    if(TEMPLATES[squad].dims == [-1, -1]):
        print(squad, "Invalid JSON! No reference dimensions given")
        exit(0)

