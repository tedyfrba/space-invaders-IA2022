# Space Invaders IA 2022

## About

Space Invaders is a two-dimensional fixed shooter game in which the player controls a ship with lasers by moving it horizontally
across the bottom of the screen and firing at descending aliens. The aim is to defeat five rows of ten aliens that move
horizontally back and forth across the screen as they advance towards the bottom of the screen. The player defeats an alien,
and earns points, by shooting it with the laser cannon. As more aliens are defeated, the aliens' movement and the game's music
both speed up.

The aliens attempt to destroy the ship by firing at it while they approach the bottom of the screen. If they reach the bottom,
the alien invasion is successful and the game ends. A special "mystery ship" will occasionally move across the top of the
screen and award bonus points if destroyed. The ship is partially protected by several stationary defense bunkers that are
gradually destroyed by projectiles from the aliens and player.

<img src="http://i.imgur.com/u2mss8o.png" width="300" height="240" />
<img src="http://i.imgur.com/mR81p5O.png" width="300" height="240"/>

## How To Play

- If you don't have [Python](https://www.python.org/downloads/) or [Pygame](http://www.pygame.org/download.shtml) installed, you can simply double click the .exe file to play the game.
  **Note:** _The .exe file needs to stay in the same directory as the sounds, images, and font folders._

- If you have the correct version of Python and Pygame installed, you can run the program in the command prompt / terminal.

```bash
python spaceinvaders.py
python main.py >> ./runs/run01.log
```

**Note:** If you're using Python 3, replace the command "python" with "python3"

## Conda Config

 ``` bash
conda activate base
conda create --name spaceInvaders01 python=3.7
conda activate spaceInvaders01
pip install pygame
pip install numpy
pip install scipy
pip install sklearn
python space_driver.py
