#!/usr/bin/env python

# Space Invaders
# Created by Lee Robinson

import pygame
import sys
from os.path import abspath, dirname
from random import choice

BASE_PATH = abspath(dirname(__file__))
FONT_PATH = BASE_PATH + '/fonts/'
IMAGE_PATH = BASE_PATH + '/images/'

# Colors (R, G, B)
WHITE = (255, 255, 255)
GREEN = (78, 255, 87)
YELLOW = (241, 255, 0)
BLUE = (80, 255, 239)
PURPLE = (203, 0, 255)
RED = (237, 28, 36)

SCREEN = pygame.display.set_mode((800, 600))
FONT = FONT_PATH + 'space_invaders.ttf'
IMG_NAMES = ['ship', 'mystery',
             'enemy1_1', 'enemy1_2',
             'enemy2_1', 'enemy2_2',
             'enemy3_1', 'enemy3_2',
             'explosionblue', 'explosiongreen', 'explosionpurple',
             'laser', 'enemylaser']
IMAGES = {name: pygame.image.load(IMAGE_PATH + '{}.png'.format(name)).convert_alpha()
          for name in IMG_NAMES}

BLOCKERS_POSITION = 450
ENEMY_DEFAULT_POSITION = 65  # Initial value for a new game
ENEMY_MOVE_DOWN = 35


class Ship(pygame.sprite.Sprite):
    def __init__(self, game):
        pygame.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(375, 540))
        self.speed = 5
        self.game = game

    def update(self, keys, *args):
        if keys[pygame.K_LEFT] and self.rect.x > 10:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.x < 740:
            self.rect.x += self.speed
        self.game.screen.blit(self.image, self.rect)

    def moveLeft(self):
        if self.rect.x > 10:
            self.rect.x -= self.speed
            self.game.screen.blit(self.image, self.rect)
            return True
        else:
            return False

    def moveRight(self):
        if self.rect.x < 740:
            self.rect.x += self.speed
            self.game.screen.blit(self.image, self.rect)
            return True
        else:
            return False


class Bullet(pygame.sprite.Sprite):
    def __init__(self, game, xpos, ypos, direction, speed, filename, side):
        pygame.sprite.Sprite.__init__(self)
        self.image = IMAGES[filename]
        self.rect = self.image.get_rect(topleft=(xpos, ypos))
        self.speed = speed
        self.direction = direction
        self.side = side
        self.filename = filename
        self.game = game

    def update(self, keys, *args):
        self.game.screen.blit(self.image, self.rect)
        self.rect.y += self.speed * self.direction
        if self.rect.y < 15 or self.rect.y > 600:
            self.kill()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, game, row, column):
        pygame.sprite.Sprite.__init__(self)
        self.row = row
        self.column = column
        self.images = []
        self.load_images()
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.game = game

    def toggle_image(self):
        self.index += 1
        if self.index >= len(self.images):
            self.index = 0
        self.image = self.images[self.index]

    def update(self, *args):
        self.game.screen.blit(self.image, self.rect)

    def load_images(self):
        images = {0: ['1_2', '1_1'],
                  1: ['2_2', '2_1'],
                  2: ['2_2', '2_1'],
                  3: ['3_1', '3_2'],
                  4: ['3_1', '3_2'],
                  }
        img1, img2 = (IMAGES['enemy{}'.format(img_num)] for img_num in
                      images[self.row])
        self.images.append(pygame.transform.scale(img1, (40, 35)))
        self.images.append(pygame.transform.scale(img2, (40, 35)))


class EnemiesGroup(pygame.sprite.Group):
    def __init__(self, game, columns, rows):
        pygame.sprite.Group.__init__(self)
        self.enemies = [[None] * columns for _ in range(rows)]
        self.columns = columns
        self.rows = rows
        self.leftAddMove = 0
        self.rightAddMove = 0
        self.moveTime = 600
        self.direction = 1
        self.rightMoves = 30
        self.leftMoves = 30
        self.moveNumber = 15
        self.timer = pygame.time.get_ticks()
        self.game = game
        self.bottom = self.game.enemyPosition + ((rows - 1) * 45) + 35
        self._aliveColumns = list(range(columns))
        self._leftAliveColumn = 0
        self._rightAliveColumn = columns - 1

    def update(self, current_time):
        if current_time - self.timer > self.moveTime:
            if self.direction == 1:
                max_move = self.rightMoves + self.rightAddMove
            else:
                max_move = self.leftMoves + self.leftAddMove

            if self.moveNumber >= max_move:
                self.leftMoves = 30 + self.rightAddMove
                self.rightMoves = 30 + self.leftAddMove
                self.direction *= -1
                self.moveNumber = 0
                self.bottom = 0
                for enemy in self:
                    enemy.rect.y += ENEMY_MOVE_DOWN
                    enemy.toggle_image()
                    if self.bottom < enemy.rect.y + 35:
                        self.bottom = enemy.rect.y + 35
            else:
                velocity = 10 if self.direction == 1 else -10
                for enemy in self:
                    enemy.rect.x += velocity
                    enemy.toggle_image()
                self.moveNumber += 1

            self.timer += self.moveTime

    def add_internal(self, *sprites):
        super(EnemiesGroup, self).add_internal(*sprites)
        for s in sprites:
            self.enemies[s.row][s.column] = s

    def remove_internal(self, *sprites):
        super(EnemiesGroup, self).remove_internal(*sprites)
        for s in sprites:
            self.kill(s)
        self.update_speed()

    def is_column_dead(self, column):
        return not any(self.enemies[row][column]
                       for row in range(self.rows))

    def random_bottom(self):
        col = choice(self._aliveColumns)
        col_enemies = (self.enemies[row - 1][col]
                       for row in range(self.rows, 0, -1))
        return next((en for en in col_enemies if en is not None), None)

    def update_speed(self):
        if len(self) == 1:
            self.moveTime = 200
        elif len(self) <= 10:
            self.moveTime = 400

    def kill(self, enemy):
        self.enemies[enemy.row][enemy.column] = None
        is_column_dead = self.is_column_dead(enemy.column)
        if is_column_dead:
            self._aliveColumns.remove(enemy.column)

        if enemy.column == self._rightAliveColumn:
            while self._rightAliveColumn > 0 and is_column_dead:
                self._rightAliveColumn -= 1
                self.rightAddMove += 5
                is_column_dead = self.is_column_dead(self._rightAliveColumn)

        elif enemy.column == self._leftAliveColumn:
            while self._leftAliveColumn < self.columns and is_column_dead:
                self._leftAliveColumn += 1
                self.leftAddMove += 5
                is_column_dead = self.is_column_dead(self._leftAliveColumn)


class Blocker(pygame.sprite.Sprite):
    def __init__(self, game, size, color, row, column):
        pygame.sprite.Sprite.__init__(self)
        self.height = size
        self.width = size
        self.color = color
        self.image = pygame.Surface((self.width, self.height))
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.row = row
        self.column = column
        self.game = game

    def update(self, keys, *args):
        self.game.screen.blit(self.image, self.rect)


class Mystery(pygame.sprite.Sprite):
    def __init__(self, game):
        pygame.sprite.Sprite.__init__(self)
        self.image = IMAGES['mystery']
        self.image = pygame.transform.scale(self.image, (75, 35))
        self.rect = self.image.get_rect(topleft=(-80, 45))
        self.row = 5
        self.moveTime = 25000
        self.direction = 1
        self.timer = pygame.time.get_ticks()
        self.playSound = True
        self.game = game

    def update(self, keys, currentTime, *args):
        resetTimer = False
        passed = currentTime - self.timer
        if passed > self.moveTime:
            if (self.rect.x < 0 or self.rect.x > 800) and self.playSound:
                self.playSound = False
            if self.rect.x < 840 and self.direction == 1:
                self.rect.x += 2
                self.game.screen.blit(self.image, self.rect)
            if self.rect.x > -100 and self.direction == -1:
                self.rect.x -= 2
                self.game.screen.blit(self.image, self.rect)

        if self.rect.x > 830:
            self.playSound = True
            self.direction = -1
            resetTimer = True
        if self.rect.x < -90:
            self.playSound = True
            self.direction = 1
            resetTimer = True
        if passed > self.moveTime and resetTimer:
            self.timer = currentTime


class EnemyExplosion(pygame.sprite.Sprite):
    def __init__(self, game, enemy, *groups):
        super(EnemyExplosion, self).__init__(*groups)
        self.image = pygame.transform.scale(self.get_image(enemy.row), (40, 35))
        self.image2 = pygame.transform.scale(self.get_image(enemy.row), (50, 45))
        self.rect = self.image.get_rect(topleft=(enemy.rect.x, enemy.rect.y))
        self.timer = pygame.time.get_ticks()
        self.game = game

    @staticmethod
    def get_image(row):
        img_colors = ['purple', 'blue', 'blue', 'green', 'green']
        return IMAGES['explosion{}'.format(img_colors[row])]

    def update(self, current_time, *args):
        passed = current_time - self.timer
        if passed <= 100:
            self.game.screen.blit(self.image, self.rect)
        elif passed <= 200:
            self.game.screen.blit(self.image2, (self.rect.x - 6, self.rect.y - 6))
        elif 400 < passed:
            self.kill()


class MysteryExplosion(pygame.sprite.Sprite):
    def __init__(self, game, mystery, score, *groups):
        super(MysteryExplosion, self).__init__(*groups)
        self.text = Text(FONT, 20, str(score), WHITE,
                         mystery.rect.x + 20, mystery.rect.y + 6)
        self.timer = pygame.time.get_ticks()
        self.game = game

    def update(self, current_time, *args):
        passed = current_time - self.timer
        if passed <= 200 or 400 < passed <= 600:
            self.text.draw(self.game.screen)
        elif 600 < passed:
            self.kill()


class ShipExplosion(pygame.sprite.Sprite):
    def __init__(self, game, ship, *groups):
        super(ShipExplosion, self).__init__(*groups)
        self.image = IMAGES['ship']
        self.rect = self.image.get_rect(topleft=(ship.rect.x, ship.rect.y))
        self.timer = pygame.time.get_ticks()
        self.game = game

    def update(self, current_time, *args):
        passed = current_time - self.timer
        if 300 < passed <= 600:
            self.game.screen.blit(self.image, self.rect)
        elif 900 < passed:
            self.kill()


class Life(pygame.sprite.Sprite):
    def __init__(self, game, xpos, ypos):
        pygame.sprite.Sprite.__init__(self)
        self.image = IMAGES['ship']
        self.image = pygame.transform.scale(self.image, (23, 23))
        self.rect = self.image.get_rect(topleft=(xpos, ypos))
        self.game = game

    def update(self, *args):
        self.game.screen.blit(self.image, self.rect)


class Text(object):
    def __init__(self, textFont, size, message, color, xpos, ypos):
        self.font = pygame.font.Font(textFont, size)
        self.surface = self.font.render(message, True, color)
        self.rect = self.surface.get_rect(topleft=(xpos, ypos))

    def draw(self, surface):
        surface.blit(self.surface, self.rect)


class SpaceInvaders(object):
    def __init__(self, clockTick):
        # It seems, in Linux buffersize=512 is not enough, use 4096 to prevent:
        #   ALSA lib pcm.c:7963:(snd_pcm_recover) underrun occurred
        pygame.mixer.pre_init(44100, -16, 1, 4096)
        pygame.init()
        self.clock = pygame.time.Clock()
        self.clockTick = clockTick
        self.caption = pygame.display.set_caption('Space Invaders')
        self.screen = SCREEN
        self.background = pygame.image.load(IMAGE_PATH + 'background.jpg').convert()
        self.startGame = False
        self.mainScreen = True
        self.gameOver = False
        # Counter for enemy starting position (increased each new round)
        self.enemyPosition = ENEMY_DEFAULT_POSITION
        self.titleText = Text(FONT, 50, 'Space Invaders', WHITE, 164, 155)
        self.titleText2 = Text(FONT, 25, 'Press any key to continue', WHITE,
                               201, 225)
        self.gameOverText = Text(FONT, 50, 'Game Over', WHITE, 250, 270)
        self.nextRoundText = Text(FONT, 50, 'Next Round', WHITE, 240, 270)
        self.enemy1Text = Text(FONT, 25, '   =   10 pts', GREEN, 368, 270)
        self.enemy2Text = Text(FONT, 25, '   =  20 pts', BLUE, 368, 320)
        self.enemy3Text = Text(FONT, 25, '   =  30 pts', PURPLE, 368, 370)
        self.enemy4Text = Text(FONT, 25, '   =  ?????', RED, 368, 420)
        self.scoreText = Text(FONT, 20, 'Score', WHITE, 5, 5)
        self.livesText = Text(FONT, 20, 'Lives ', WHITE, 640, 5)

        self.life1 = Life(self, 715, 3)
        self.life2 = Life(self, 742, 3)
        self.life3 = Life(self, 769, 3)
        self.livesGroup = pygame.sprite.Group(self.life1, self.life2, self.life3)

    def reset(self, score):
        self.player = Ship(self)
        self.playerGroup = pygame.sprite.Group(self.player)
        self.explosionsGroup = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.mysteryShip = Mystery(self)
        self.mysteryGroup = pygame.sprite.Group(self.mysteryShip)
        self.enemyBullets = pygame.sprite.Group()
        self.make_enemies()
        self.allSprites = pygame.sprite.Group(self.player, self.enemies,
                                       self.livesGroup, self.mysteryShip)
        self.keys = pygame.key.get_pressed()

        self.timer = pygame.time.get_ticks()
        self.noteTimer = pygame.time.get_ticks()
        self.shipTimer = pygame.time.get_ticks()
        self.score = score
        self.makeNewShip = False
        self.shipAlive = True

    def make_blockers(self, number):
        blockerGroup = pygame.sprite.Group()
        for row in range(4):
            for column in range(9):
                blocker = Blocker(self, 10, GREEN, row, column)
                blocker.rect.x = 50 + (200 * number) + (column * blocker.width)
                blocker.rect.y = BLOCKERS_POSITION + (row * blocker.height)
                blockerGroup.add(blocker)
        return blockerGroup

    @staticmethod
    def should_exit(evt):
        # type: (pygame.event.EventType) -> bool
        return evt.type == pygame.QUIT or (evt.type == pygame.KEYUP and evt.key == pygame.K_ESCAPE)

    def check_input(self):
        self.keys = pygame.key.get_pressed()
        for e in pygame.event.get():
            if self.should_exit(e):
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.make_shot()

    def make_shot(self):
        if len(self.bullets) == 0 and self.shipAlive:
            if self.score < 1000:
                bullet = Bullet(self, self.player.rect.x + 23,
                                self.player.rect.y + 5, -1,
                                15, 'laser', 'center')
                self.bullets.add(bullet)
                self.allSprites.add(self.bullets)
            else:
                leftbullet = Bullet(self, self.player.rect.x + 8,
                                    self.player.rect.y + 5, -1,
                                    15, 'laser', 'left')
                rightbullet = Bullet(self, self.player.rect.x + 38,
                                     self.player.rect.y + 5, -1,
                                     15, 'laser', 'right')
                self.bullets.add(leftbullet)
                self.bullets.add(rightbullet)
                self.allSprites.add(self.bullets)

    def make_enemies(self):
        enemies = EnemiesGroup(self, 10, 5)
        for row in range(5):
            for column in range(10):
                enemy = Enemy(self, row, column)
                enemy.rect.x = 157 + (column * 50)
                enemy.rect.y = self.enemyPosition + (row * 45)
                enemies.add(enemy)

        self.enemies = enemies

    def make_enemies_shoot(self):
        if (pygame.time.get_ticks() - self.timer) > 700 and self.enemies:
            enemy = self.enemies.random_bottom()
            self.enemyBullets.add(
                Bullet(self, enemy.rect.x + 14, enemy.rect.y + 20, 1, 5,
                       'enemylaser', 'center'))
            self.allSprites.add(self.enemyBullets)
            self.timer = pygame.time.get_ticks()

    def calculate_score(self, row):
        scores = {0: 30,
                  1: 20,
                  2: 20,
                  3: 10,
                  4: 10,
                  5: choice([50, 100, 150, 300])
                  }

        score = scores[row]
        self.score += score
        return score

    def create_main_menu(self):
        self.enemy1 = IMAGES['enemy3_1']
        self.enemy1 = pygame.transform.scale(self.enemy1, (40, 40))
        self.enemy2 = IMAGES['enemy2_2']
        self.enemy2 = pygame.transform.scale(self.enemy2, (40, 40))
        self.enemy3 = IMAGES['enemy1_2']
        self.enemy3 = pygame.transform.scale(self.enemy3, (40, 40))
        self.enemy4 = IMAGES['mystery']
        self.enemy4 = pygame.transform.scale(self.enemy4, (80, 40))
        self.screen.blit(self.enemy1, (318, 270))
        self.screen.blit(self.enemy2, (318, 320))
        self.screen.blit(self.enemy3, (318, 370))
        self.screen.blit(self.enemy4, (299, 420))

    def check_collisions(self):
        pygame.sprite.groupcollide(self.bullets, self.enemyBullets, True, True)

        for enemy in pygame.sprite.groupcollide(self.enemies, self.bullets,
                                         True, True).keys():
            self.calculate_score(enemy.row)
            EnemyExplosion(self, enemy, self.explosionsGroup)
            self.gameTimer = pygame.time.get_ticks()

        for mystery in pygame.sprite.groupcollide(self.mysteryGroup, self.bullets,
                                           True, True).keys():
            score = self.calculate_score(mystery.row)
            MysteryExplosion(self, mystery, score, self.explosionsGroup)
            newShip = Mystery(self)
            self.allSprites.add(newShip)
            self.mysteryGroup.add(newShip)

        for player in pygame.sprite.groupcollide(self.playerGroup, self.enemyBullets,
                                          True, True).keys():
            if self.life3.alive():
                self.life3.kill()
            elif self.life2.alive():
                self.life2.kill()
            elif self.life1.alive():
                self.life1.kill()
            else:
                self.gameOver = True
                self.startGame = False
            ShipExplosion(self, player, self.explosionsGroup)
            self.makeNewShip = True
            self.shipTimer = pygame.time.get_ticks()
            self.shipAlive = False

        if self.enemies.bottom >= 540:
            pygame.sprite.groupcollide(self.enemies, self.playerGroup, True, True)
            if not self.player.alive() or self.enemies.bottom >= 600:
                self.gameOver = True
                self.startGame = False

        pygame.sprite.groupcollide(self.bullets, self.allBlockers, True, True)
        pygame.sprite.groupcollide(self.enemyBullets, self.allBlockers, True, True)
        if self.enemies.bottom >= BLOCKERS_POSITION:
            pygame.sprite.groupcollide(self.enemies, self.allBlockers, False, True)

    def create_new_ship(self, createShip, currentTime):
        if createShip and (currentTime - self.shipTimer > 900):
            self.player = Ship(self)
            self.allSprites.add(self.player)
            self.playerGroup.add(self.player)
            self.makeNewShip = False
            self.shipAlive = True

    def create_game_over(self, currentTime):
        self.screen.blit(self.background, (0, 0))
        passed = currentTime - self.timer
        if passed < 750:
            self.gameOverText.draw(self.screen)
        elif 750 < passed < 1500:
            self.screen.blit(self.background, (0, 0))
        elif 1500 < passed < 2250:
            self.gameOverText.draw(self.screen)
        elif 2250 < passed < 2750:
            self.screen.blit(self.background, (0, 0))
        elif passed > 3000:
            self.mainScreen = True

        for e in pygame.event.get():
            if self.should_exit(e):
                sys.exit()

    def main(self):
        while True:
            if self.mainScreen:
                self.screen.blit(self.background, (0, 0))
                self.titleText.draw(self.screen)
                self.titleText2.draw(self.screen)
                self.enemy1Text.draw(self.screen)
                self.enemy2Text.draw(self.screen)
                self.enemy3Text.draw(self.screen)
                self.enemy4Text.draw(self.screen)
                self.create_main_menu()
                for e in pygame.event.get():
                    if self.should_exit(e):
                        sys.exit()
                    if e.type == pygame.KEYUP:
                        # Only create blockers on a new game, not a new round
                        self.allBlockers = pygame.sprite.Group(self.make_blockers(0),
                                                        self.make_blockers(1),
                                                        self.make_blockers(2),
                                                        self.make_blockers(3))
                        self.livesGroup.add(self.life1, self.life2, self.life3)
                        self.reset(0)
                        self.startGame = True
                        self.mainScreen = False

            elif self.startGame:
                self.run_game()

            elif self.gameOver:
                currentTime = pygame.time.get_ticks()
                # Reset enemy starting position
                self.enemyPosition = ENEMY_DEFAULT_POSITION
                self.create_game_over(currentTime)

            pygame.display.update()
            self.clock.tick(self.clockTick)

    def setup_game(self):
        self.screen.blit(self.background, (0, 0))
        self.titleText.draw(self.screen)
        self.titleText2.draw(self.screen)
        self.enemy1Text.draw(self.screen)
        self.enemy2Text.draw(self.screen)
        self.enemy3Text.draw(self.screen)
        self.enemy4Text.draw(self.screen)

        self.allBlockers = pygame.sprite.Group(self.make_blockers(0),
                                        self.make_blockers(1),
                                        self.make_blockers(2),
                                        self.make_blockers(3))
        self.livesGroup.add(self.life1, self.life2, self.life3)
        self.reset(0)

    def run_game(self):
        if not self.enemies and not self.explosionsGroup:
            currentTime = pygame.time.get_ticks()
            if currentTime - self.gameTimer < 3000:
                self.screen.blit(self.background, (0, 0))
                self.scoreText2 = Text(FONT, 20, str(self.score),
                                       GREEN, 85, 5)
                self.scoreText.draw(self.screen)
                self.scoreText2.draw(self.screen)
                self.nextRoundText.draw(self.screen)
                self.livesText.draw(self.screen)
                self.livesGroup.update()
                self.check_input()
            if currentTime - self.gameTimer > 3000:
                # Move enemies closer to bottom
                self.enemyPosition += ENEMY_MOVE_DOWN
                self.reset(self.score)
                self.gameTimer += 3000
        else: # a new stage
            currentTime = pygame.time.get_ticks()
            self.screen.blit(self.background, (0, 0))
            self.allBlockers.update(self.screen)
            self.scoreText2 = Text(FONT, 20, str(self.score), GREEN,
                                   85, 5)
            self.scoreText.draw(self.screen)
            self.scoreText2.draw(self.screen)
            self.livesText.draw(self.screen)
            self.check_input()
            self.enemies.update(currentTime)
            self.allSprites.update(self.keys, currentTime)
            self.explosionsGroup.update(currentTime)
            self.check_collisions()
            self.create_new_ship(self.makeNewShip, currentTime)
            self.make_enemies_shoot()

        game_info = GameInfo(self.shipAlive, self.score)

        return game_info


class GameInfo:
    def __init__(self, shipAlive, score):
        self.shipAlive = shipAlive
        self.score = score

if __name__ == '__main__':
    gm = SpaceInvaders(0)
    gm.main()
