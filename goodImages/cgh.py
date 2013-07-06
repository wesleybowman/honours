'''
    cgh allows the user to created Computer Generated Holograms out of an
    image.
    Copyright (C) 2013 Wesley A. Bowman

    This program is free software; you can redistribute it and/or modify it 
    under the terms of the GNU General Public License as published by the 
    Free Software Foundation; either version 2 of the License, 
    or (at your option) any later version.

    This program is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
    See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License 
    along with this program; if not, write to the 
    Free Software Foundation, Inc., 59 Temple Place, Suite 330, 
    Boston, MA 02111-1307 USA
    
    This program takes any input images, and turns all of the pixels below
    a certain threshold (in this case 0.5, but all images are normalized to 
    be between zero and one), and turns them into point sources, which then
    propagate to the hologram, and interfere with one another.

'''

import numpy as np
import matplotlib.pyplot as plt
import tables
import time

import sys
import numexpr as ne

def getImage(image):
    '''Get the initial image that one wants to make a CGH out of. '''
    img=plt.imread(image)
    
    return img

def normalizeImage(img):
    '''This is to make sure every input image is seen as the same by the
       program later on. This way, the image will be between 0 and 1.'''
    
    imgMax=np.amax(img)
    
    for (i,j),k in np.ndenumerate(img):
        img[i,j]=k/imgMax
    
    return img

def getParameters(a,b,c=1,d=0, e=0):
    '''Get initial parameters.
       a is wavelength in meters
       b is how many dots per inch is needed
       c is the dimensions of the hologram in inches
       d is the offset of the object points on the x-axis
       e is the offset of the object points on the y-axis'''
       
    wavelength = a
    sampling=0.0254/b #0.0254 meters is one inch
    dimensions = c
    xOffset = d 
    yOffset = e
    
    return wavelength,sampling,dimensions,xOffset,yOffset

def constants():
    '''Defines some values that do not need to be modified. '''
        
    k=2*np.pi/wavelength

    holoRange = dimensions*0.0254/2
    ipx = np.arange((-1*holoRange),(holoRange+sampling),sampling)
    ipy = np.arange((-1*holoRange),(holoRange+sampling),sampling)
    
    ipx=np.reshape(ipx,(1,ipx.shape[0]))
    ipy=np.reshape(ipy,(1,ipy.shape[0]))
    
    ipxShape=ipx.shape[1]
    ipyShape=ipy.shape[1]
    
    hologram = np.zeros((ipxShape,ipyShape))+0j
    
    return k,holoRange,ipx,ipy,ipxShape,ipyShape,hologram

def getHoloPara(a=0.03,b=0.03,c=2):
    '''Get the parameters needed for creating the hologram.
       a is width in meters
       b is height in meters
       c is depth in meters'''
       
    width = a
    height = b
    depth = c
        
    return width,height,depth
    
def getObjectpoints(img, width, height, depth):
    '''Not vectorized, and no longer used. Here as a reference.'''
    
    obj = np.double(img)
    objX=obj.shape[0]
    objY=obj.shape[1]
    
    thresh = 0.5
    row = 0 

    for (i,j),value in np.ndenumerate(obj):

        if obj[i,j]<thresh:            
            if row==0:
                objectpoints=np.array([[(i-objX/2)*(width/objX),
                          (j-objY/2)*(height/objY),
                            depth]])
                row+=1
            else:
                temp= np.array([[(i-objX/2)*(width/objX),
                      (j-objY/2)*(height/objY),
                        depth]])
                objectpoints=np.vstack((objectpoints,temp))
            
            
                
    return objectpoints
    
def vec_get_objectpoints(img, width, height, depth):
    '''Figure out which points in the image will be used as source points. '''
        
    obj = np.double(img)
    objX=obj.shape[0]
    objY=obj.shape[1]
    
    thresh = 0.5
    
    w=(width/objX)
    h=(height/objY)
    
    index=np.where(obj>thresh)
    
    xArray=(index[0]-objX/2)*w
    yArray=(index[1]-objY/2)*h
    zArray=depth*np.ones((xArray.shape[0],1))
    
    xArray=np.reshape(xArray,(xArray.shape[0],1))
    yArray=np.reshape(yArray,(yArray.shape[0],1))
    
    objectpoints=np.hstack((xArray,yArray,zArray))
    
    return objectpoints

def getComplexwave():
    '''Not vectorized, and no longer used. Here as a reference.'''
    for o in xrange(objPointShape):
        print o+1

        for i in xrange(ipxShape):
            for j in xrange(ipyShape):
                dx=objectpoints[o,0] - ipx[0,i]
                dy=objectpoints[o,1] - ipy[0,j]
                dz=objectpoints[o,2]
                
                distance=np.sqrt(dx**2+dy**2+dz**2)
                complexwave=np.exp(1j*k*distance)
                
                hologram[i,j]=hologram[i,j]+complexwave

def vec_get_complexwave(objectpoints):
    '''Calculate the complex wave due to each object point, then add all 
       of the complex waves together to find the interference pattern on
       the hologram.'''
    
    dx = objectpoints[:, 0, None] - ipx # shape (objPointShape, ipxShape)
    dy = objectpoints[:, 1, None] - ipy # shape (objPointShape, ipyShape)
    dz = objectpoints[:, 2, None] # shape (objPointShape, 1)
    
  
    x=(dx*dx)[...,None]
    y=(dy*dy)[:,None,:]    
    z=(dz*dz)[...,None] 
    print sys.getsizeof(x),sys.getsizeof(y),sys.getsizeof(z)
    
    x=data.createArray('/','x',x)
    y=data.createArray('/','y',y)
    z=data.createArray('/','z',z)
    
#    expr=tables.Expr('x+y+z')    
#    
#    r=data.createArray('/','r',x)
#    expr.setOutput(r)
    
#    dx.dump('dx.dat')
#    dy.dump('dy.dat')
#    dz.dump('dz.dat')
    
    try:
        distance = np.sqrt((dx*dx)[..., None] + 
                       (dy*dy)[:, None, :] +
                       (dz*dz)[..., None])
    
#    test=ne.evaluate((dx*dx)[..., None] + (dy*dy)[:, None, :] )
#    +(dz*dz)[..., None])
#    distance=ne.evaluate(np.sqrt((dx*dx)[..., None] + 
#                       (dy*dy)[:, None, :] +
#                       (dz*dz)[..., None]))
                       
#    distance=data.createArray('/','distance',np.sqrt(x+y+z))
#    print test.eval()
#    distance=np.sqrt(x)
#    distance=np.sqrt(xyz)
                       
    except: 
        raise exception
        return dx,dy,dz
    complexwave = np.exp(1j*k*distance)
    return complexwave.sum(axis=0)

def plotHologram(hologram):
    '''Plotting the real part of the hologram, and using a grey scale color
       map'''
    
    plt.imshow(hologram.real,cmap=plt.cm.Greys_r)
    plt.colorbar()
    plt.show()

if __name__=='__main__':
    
    t1=time.time()
    
#    img=getImage('smallA.png')
    img=getImage('object.png')
#    img=getImage('dot.png')


    img=np.mean(img,2)
    
    img=normalizeImage(img)
    
    wavelength,sampling,dimensions,xOffset,yOffset=getParameters(632e-9,600,1)
    
    k,holoRange,ipx,ipy,ipxShape,ipyShape,hologram=constants()
        
    width,height,depth=getHoloPara() #left with default args
        
    objectpoints=vec_get_objectpoints(img, width, height, depth)
        

    '''To test the 3-D portion of this program, two images can be stacked 
       at different depth values, so that when reconstructed, you can get
       two distinct images at different z values. To stack two images,
       use the following snippit of code.'''    
#    
#    img2=plt.imread('smallB.png')
#    img2=np.mean(img2,2)
#    depth2=4
#    
#    objectpoints2=vec_get_objectpoints(img2, width, height, depth2)
#    
#    objectpoints=np.vstack((objectpoints,objectpoints2))
#
    objPointShape=objectpoints.shape[0]    
    
    #offset the x-axis by some amount
    objectpoints[:,1] = objectpoints[:,1]+xOffset
    #offset the y-axis by some amount
    objectpoints[:,0] = objectpoints[:,0]+yOffset
    
    print 'Hologram resolution = %d,%d \n' %(ipxShape,ipyShape)
    print 'Number of source points from image = %d \n' %objPointShape
    print 'Calculating hologram for source points:'
    
    with tables.openFile('data.h5','w') as data:
    
        hologram=vec_get_complexwave(objectpoints)
    
    print hologram
#    plt.imsave('output.png',hologram.real,cmap=plt.cm.Greys_r)

    t2=time.time()
    print t2-t1
    
#    plotHologram(hologram)