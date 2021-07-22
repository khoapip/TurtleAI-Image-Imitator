from neat import nn, population
import cv2
import neat
from LSTurtleWorld import LSTurtleWorld
import numpy as np
import visualize
import pickle


#  global img
#  mona = cv2.imread('mona.jpg', cv2.CV_8UC1)
#  mona = cv2.resize(mona, (20, 20))
#  _, mona = cv2.threshold(mona, 127, 255, cv2.THRESH_BINARY )
#  mona = mona/255
#  img = mona

def eval_fitness(genomes, config):
    best_instance = None
    genome_number = 0

    bestdiff  = -np.Inf
    best_correct = 0

    for j, (i, g) in enumerate(genomes):
        #  net = nn.FeedForwardNetwork.create(g, config)
        #  net = nn.recurrent.RecurrentNetwork.create(g, config)
        net = nn.FeedForwardNetwork.create(g, config)
        img = np.random.randint(0,2,size = (20,20))
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

        while True:
            #  print(LSW.ImageDiff(img))

            if diff == 0:
                g.fitness += 50000
                break

            if steps < 0: 
                break
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



            if instruction != 4 and LSW.position()[0] == prev_pos[0] and LSW.position()[1] == prev_pos[1]:
                g.fitness -= 10
                break

            if LSW.nearestDist() <= prev_dist:
                g.fitness += 1.5
            else:
                g.fitness -= 1.5

            if diff > LSW.ImageDiff(img):
                break

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
    

            


            
if __name__ == "__main__":
    #  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            #  neat.DefaultSpeciesSet, neat.DefaultStagnation,
            #  'config/config-recurrent.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            'config/config-feedforward.txt')

    pop = population.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(100))
    winner = pop.run(eval_fitness, 3000)
    with open('best/winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

    for i in range(len(stats.best_genomes(10))):
        with open('best/top{}'.format(i), 'wb') as f:
            pickle.dump(stats.best_genomes(10)[i], f)


    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    




