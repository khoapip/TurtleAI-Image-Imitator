import numpy as np
import cv2
import pickle
import neat
from neat import nn
import argparse
from skimage.io import imsave
from LSTurtleWorldExtend import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default = 'picture/mona.jpg', help='img path')
    parser.add_argument('--out', default ='output', help='output folder')
    parser.add_argument('--genome', default ='checkpoint/best-feedforward/top1', help='link config')
    parser.add_argument('--config', default ='config/config-feedforward.txt', help='link config')
    args = parser.parse_args()
    return args

def eval_fitness(genomes,img, config):
    best_instance = None
    genome_number = 0

    bestdiff  = -np.Inf
    best_correct = 0

    for j, (i, g) in enumerate(genomes):
        #  net = nn.FeedForwardNetwork.create(g, config)
        net = nn.FeedForwardNetwork.create(g, config)
        steps = 100
        correct = 0
        g.fitness = -100
        LSW = LSTurtleWorld(20,20)
        LSW.evaluateClosest(img)
        prev_pos = LSW.position()
        diff = LSW.ImageDiff(img)
        prev_dist = LSW.nearestDist() 
        press = 0 
        move = []
        fitness = []
        pos = []

        while diff != 0 :

            if diff == 0:
                g.fitness += 50000
                break

            #  if steps < 0: 
                #  #  break
            inputs = LSW.getdata(img)
            outputs = net.activate(inputs)
            instruction = outputs.index(max(outputs))
            move.append(instruction)

            if instruction == 0:
                LSW.InterpretString('u')

            elif instruction == 1:
                LSW.InterpretString('d')

            elif instruction == 2:
                LSW.InterpretString('l')

            elif instruction == 3:
                LSW.InterpretString('r')
            elif instruction == 4:
                press += 1
                #  score -= 100
                g.fitness -= 5
                LSW.InterpretString('.')



            #  if instruction != 4 and LSW.position()[0] == prev_pos[0] and LSW.position()[1] == prev_pos[1]:
                #  g.fitness -= 10
                #  break

            if LSW.nearestDist() <= prev_dist:
                g.fitness += 1.5
            else:
                g.fitness -= 1.5
            #  if diff > LSW.ImageDiff(img):
                #  break


            if LSW.grid[LSW.nearest[0]][LSW.nearest[1]] == img[LSW.nearest[0]][LSW.nearest[1]]: 
                correct+= 1 
                g.fitness += 100
                steps = 100
                LSW.evaluateClosest(img)
                diff = LSW.ImageDiff(img)

            if LSW.nearest is not None: 
                prev_dist = LSW.nearestDist()
            fitness += [g.fitness]
            prev_pos = LSW.position()
            pos += [prev_pos]
            steps -= 1

        if best_correct < correct:
            bestdiff = LSW.ImageDiff(img)
            best_correct = correct
    
    
    print(bestdiff, " ", best_correct)
    return LSW


if __name__ == "__main__":
    args =  parse_args()
    if args.img == 'random':
        img = np.random.randint(0,2,size=(20,20))
        print('benchmark is saved in: output/benchmark.jpg')
        imsave('output/benchmark.jpg', img)

    else:
        img = cv2.imread(args.img, cv2.CV_8UC1)
        img = cv2.resize(img, (20, 20))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY )
        img = img/255
        print('benchmark is saved in: output/benchmark.jpg')
        imsave('output/benchmark.jpg', img)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,args.config)
    with open(args.genome, "rb") as f:
        genome = pickle.load(f)
    genomes = [(1, genome)]
    LSW = eval_fitness(genomes,img, config)
    LSW.animate()
    LSW.save()


