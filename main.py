# https://neat-python.readthedocs.io/en/latest/xor_example.html
from spaceinvaders import SpaceInvaders
import pygame
import neat
import os
import time
import pickle


class PlayGame:
    def __init__(self, genome):
        self.genome = genome
        self.game = SpaceInvaders(0)
        self.ship = None
        self.enemyBullets = None

    def test_ai(self, net):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((self.right_paddle.y, abs(
                self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            if decision == 1:  # AI moves up
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:  # AI moves down
                self.game.move_paddle(left=False, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome, config):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play to determine their fitness.
        """
        run = True
        start_time = time.time()

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        self.game.setup_game()
        self.ship = self.game.player
        self.enemyBullets = self.game.enemyBullets

        while run:
            shipAlive = True
            while shipAlive:
                game_info = self.game.run_game()

                # shooting all the time
                self.game.make_shot()
                self.move_ai_ship(net)

            # if draw:
            #     self.game.draw(draw_score=False, draw_hits=True)

                pygame.display.update()
            # self.game.clock.tick(self.game.clockTick)

                duration = time.time() - start_time

                # if game_info.score == 1 or game_info.shipAlive:
                if game_info.shipAlive:
                    self.calculate_fitness(game_info, duration)
                    break

                shipAlive = game_info.shipAlive

        return False

    def calculate_fitness(self, game_info, duration):
        self.genome.fitness += game_info.hits + duration


    def move_ai_ship(self, net):
        """
        Determine where to move left and right ship based on the
        neural networks that control them and avoid the Bullets.
        """
        decision = 0
        for (bullet) in self.enemyBullets:
            # print(self.ship.rect)
            # print(bullet.rect)
            # print(abs(self.ship.rect.x - bullet.rect.x))
            output = net.activate(
                (self.ship.rect.x, abs(self.ship.rect.x - bullet.rect.x), bullet.rect.y))
            decision = output.index(max(output))

        # print(decision)
        valid = True
        if decision == 0:  # Don't move
            self.genome.fitness -= 0.01  # we want to discourage this
        elif decision == 1 and self.ship.rect.x > 10:  # Move left
            valid = self.ship.moveLeft()
        elif self.ship.rect.x < 740:  # Move right
            valid = self.ship.moveRight()

        if not valid:  # If the movement makes the paddle go off the screen punish the AI
            self.genome.fitness -= 1


def eval_genomes(genomes, config):

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0

        game = PlayGame(genome)

        force_quit = game.train_ai(genome, config)
        if force_quit:
            quit()


def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 20)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    spcinvdrs = PlayGame()
    pygame.display.set_caption("SpcInvdrs")
    spcinvdrs.test_ai(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neatConfig.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)
    # test_best_network(config)
