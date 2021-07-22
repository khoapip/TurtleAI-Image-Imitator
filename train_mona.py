from neat import nn, population
import cv2
import neat
from LSTurtleWorld import LSTurtleWorld
import numpy as np
import visualize
import pickle


global img
mona = cv2.imread('picture/mona.jpg', cv2.CV_8UC1)
mona = cv2.resize(mona, (20, 20))
_, mona = cv2.threshold(mona, 127, 255, cv2.THRESH_BINARY )
mona = mona/255
img = mona

def eval_fitness(genomes, config):
    best_instance = None
    genome_number = 0

    bestdiff  = -np.Inf

    for j, (i, g) in enumerate(genomes):
        #  net = nn.FeedForwardNetwork.create(g, config)
        net = nn.recurrent.RecurrentNetwork.create(g, config)

        steps = 100
        g.fitness = -100
        LSW = LSTurtleWorld(20,20)
        LSW.evaluateRandom(img)
        prev_pos = LSW.position()
        diff = LSW.ImageDiff(img)
        press = 0 
        fitness = []
        pos = []
        no_step = 0

        while True:
            
            if diff == 0:
                g.fitness += 50000
                break

            if steps < 0: 
                break
            inputs = LSW.getdata(img)[2:]
            inputs = [no_step] + inputs
            outputs = net.activate(inputs)
            instruction = outputs.index(max(outputs))

            if instruction == 0:
                LSW.InterpretString('u')

            elif instruction == 1:
                LSW.InterpretString('d')

            elif instruction == 2:
                LSW.InterpretString('l')

            elif instruction == 3:
                LSW.InterpretString('r')

            elif instruction == 4:
                LSW.InterpretString('.')

            if instruction != 4 and LSW.position()[0] == prev_pos[0] and LSW.position()[1] == prev_pos[1]:
                g.fitness -= 5
                break


            if diff > LSW.ImageDiff(img):
                g.fitness -= 10
                break
            elif diff < LSW.ImageDiff(img):
                g.fitness += 30
                steps = 100
                diff = LSW.ImageDiff(img)

            prev_pos = LSW.position()
            steps -= 1
            no_step += 1

        if LSW.ImageDiff(img) > bestdiff:
            bestdiff = LSW.ImageDiff(img)
    
    print(bestdiff)

            


            
if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            'config/config-mona-recurrent.txt')

    pop = population.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(100))
    winner = pop.run(eval_fitness, 5000)
    with open('best_mona/winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

    for i in range(len(stats.best_genomes(10))):
        with open('checkpoint/best_mona/top{}'.format(i), 'wb') as f:
            pickle.dump(stats.best_genomes(10)[i], f)


    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    




