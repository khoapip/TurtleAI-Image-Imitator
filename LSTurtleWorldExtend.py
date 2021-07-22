import random
import math
from neat import nn, population
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imsave
import cv2
from my_numpy_turtle_extend import Turtle

class LSTurtleWorld:
    def __init__(self,rows,cols):
        self.rows = rows
        self.nearest = None
        self.cols= cols
        self.grid = np.random.randint(0,2,size= (20,20)) != 0
        self.turtle = Turtle(self.grid,deg=True)
        self.dx = 1
        self.da = 90
        self.Map = {'f':self.f,'r':self.right,'l':self.left, '[':self.push,
                ']':self.pop, 'u': self.up, 'd': self.down, '.': self.dot}
        self.state = [self.grid.copy()]
        self.u()

    def save(self):
        print('Result is saved in: ouput/out.jpg ')
        imsave('output/out.jpg',self.grid)


    def animate(self):
        plt.show()
        fig = plt.figure()
        im = plt.imshow(self.state[0], cmap = 'gist_gray_r', vmin =0 , vmax =1 )
        im.set_data(self.state[0])
        for i in range(1,len(self.state)):
            state = self.state[i]
            im.set_data(state)
            plt.pause(0.1)
            #  print(1)
            plt.draw()
        plt.pause(5)

    def random_distance(self, image = None):
        diff = np.abs(image - self.grid)
        mask = (np.where(diff == 1))
        mask = np.concatenate((mask[0][np.newaxis].T, mask[1][np.newaxis].T), axis =1)
        if mask.shape[0] == 0:
            return None
        indx = np.random.randint(mask.shape[0])

        return mask[indx]

    def closest_distance(self, image = None):
        diff = np.abs(image - self.grid)
        mask = (np.where(diff == 1))
        mask = np.concatenate((mask[0][np.newaxis].T, mask[1][np.newaxis].T), axis =1)
        if mask.shape[0] == 0:
            return None,None
        index = np.array([self.position()[0], self.position()[1]])
        dist_matrix  =  np.sqrt(np.sum((index - mask) **2 , axis =1 ))
        min_indx = mask[np.argmin(dist_matrix)]

        return min_indx, np.min(dist_matrix)

    def evaluateRandom(self, image):
        diff = np.abs(self.grid - image)
        if self.nearest is None:
            self.nearest = self.random_distance(image)
        else:
            if diff[int(self.nearest[0])][int(self.nearest[1])] == 0:
                self.nearest = self.random_distance(image)

    def evaluateClosest(self, image):
        diff = np.abs(self.grid - image)
        if self.nearest is None:
            self.nearest = self.closest_distance(image)[0]

        else:
            if diff[int(self.nearest[0])][int(self.nearest[1])] == 0:
                self.nearest = self.closest_distance(image)[0]

    def nearestDist(self):
        return np.sqrt(np.sum((self.position() - self.nearest)**2))

    def getdata(self, img):
        pos = self.turtle.position
        return [pos[0] - self.nearest[0], pos[1] - self.nearest[1], pos[0], pos[1], self.grid.shape[0] - pos[0] -1, self.grid.shape[1] - pos[1] - 1]

    def position(self):
        pos = np.array([int(self.turtle.position[0]), int(self.turtle.position[1])])
        if pos[0] < 0:
            pos[0] = 0
        if pos[1] < 0:
            pos[1] = 0
        if pos[0] >= self.grid.shape[0]: pos[0] = self.grid.shape[0]-1
        if pos[1] >= self.grid.shape[1]:
            pos[1] = self.grid.shape[1]-1 
        return pos

    def dot(self):
        self.turtle.dot()
        self.state += [(self.grid).copy()]

    def print(self):
        grid = self.grid.astype(int)
        print(grid)
        return

    def u(self):
        self.turtle.penup()

    def d(self):
        self.turtle.pendown() 

    def f(self):
        self.turtle.forward(self.dx)
        state = (self.grid).copy().astype('float')
        state[int(self.position()[0])][int(self.position()[1])] = 0.3
        self.state += [state]
        return

    def down(self):
        self.turtle.forward(self.dx)
        state = (self.grid).copy().astype('float')
        state[int(self.position()[0])][int(self.position()[1])] = 0.3
        self.state += [state]

    def up(self):
        self.turtle.rotate(-1*self.da)
        self.turtle.rotate(-1*self.da)
        self.turtle.forward(self.dx)
        self.turtle.rotate(-1*self.da)
        self.turtle.rotate(-1*self.da)
        state = (self.grid).copy().astype('float')
        state[int(self.position()[0])][int(self.position()[1])] = 0.3
        self.state += [state]

    def left(self):
        self.turtle.rotate(-1*self.da)
        self.turtle.forward(self.dx)
        self.turtle.rotate(self.da)
        state = (self.grid).copy().astype('float')
        state[int(self.position()[0])][int(self.position()[1])] = 0.3
        self.state += [state]
        
    def right(self):
        self.turtle.rotate(self.da)
        self.turtle.forward(self.dx)
        self.turtle.rotate(-1*self.da)
        state = (self.grid).copy().astype('float')
        state[int(self.position()[0])][int(self.position()[1])] = 0.3
        self.state += [state]
    
    def r(self):
        self.turtle.rotate(-1*self.da)
        return

    def l(self):
        self.turtle.rotate(self.da)
        return

    def push(self):
        self.turtle.push()
        self.state += [(self.grid.copy())]
        return

    def pop(self):
        if self.turtle.stacklen() != 0:
            self.turtle.pop()
        self.state+= [(self.grid.copy())]
        return

    def reset(self):
        self.grid = np.zeros((self.rows,self.cols),'bool')
        self.turtle = Turtle(self.grid,deg=True)
        self.state = [self.grid]

    def InterpretString(self,commandstring):
        for l in commandstring:
            if l in self.Map:
                #print(l)
                fn = self.Map[l]
                fn()

    def ImageDiff(self, image):
        #  print(image)
        return -np.sum(np.abs(self.grid - image))

    def CountOnes(self):
        return np.sum(self.grid)


    def testState(self):
        print(self.state)


if __name__ == "__main__":
    #  mona = cv2.imread('mona.jpg', cv2.CV_8UC1)
    #  mona = cv2.resize(mona, (20, 20))
    #  _, mona = cv2.threshold(mona, 127, 255, cv2.THRESH_BINARY )
    #  #  cv2.imwrite('mona2bit.jpg', mona)
    #  mona = mona/255
    mona  = np.zeros((20,20))

    LSW = LSTurtleWorld(20,20)
    #  LSW.grid = np.random.randint(0,2,(20,20))
    #  LSW.grid = np.zeros((20,20))
    LSW.InterpretString('.........')

    #  print(LSW.ImageDiff(mona))
    #  LSW.grid = mona
    #  img = np.zeros((20,20))
    #  LSW.evaluateClosest(img)
    #  LSW.state =[]
    #  while(LSW.nearest is not None):
        #  LSW.state += [img.copy()]
        #  img[LSW.nearest[0]][LSW.nearest[1]] = 1
        #  LSW.evaluateClosest(img)

    LSW.animate()
