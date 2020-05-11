import random
import numpy as np
import matplotlib.pyplot as plt
import pygame
import pymunk
from pymunk import pygame_util
from pymunk.vec2d import Vec2d
from random import randint
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

ANGLE = 3

screen_size = width, height = 720, 420
screen = pygame.display.set_mode(screen_size)
screen.set_alpha(None)
options = pymunk.pygame_util.DrawOptions(screen)
black = 0, 0, 0
red = 255, 80, 80
green = 128, 255, 128
yellow = 255, 255, 102
blue = 25, 230, 179
dt = 1./60
space = pymunk.Space()
space.gravity = 0, 0

class Score():
    def __init__(self):
        self.score_total = []
        self.game = []
        self.av = []
        self.score = np.zeros(50)

    def record(self, game, score):
        self.score_total.append(score)
        self.game.append(game)
        self.score[game%50] = score
        if (game+1)%50 == 0:
            self.av.append(np.sum(self.score)/50)

    def plot(self):
        x = np.arange(len(self.av))
        plt.plot(x, self.av, color = "black")
        plt.title("Avarage score over time")
        plt.xlabel("avaraged at")
        plt.ylabel("score")
        plt.show()

        plt.plot(self.game, self.score_total, color="black")
        plt.title("Score evolution with time")
        plt.xlabel("game")
        plt.ylabel("score")
        plt.show()

class PositionHandler():
    def __init__(self, obstacles, distance):
        self.obstacles = obstacles
        self.distance = distance

    def place_reward(self, x, y):
        for obstacle in self.obstacles:
            c = obstacle.body.position
            dist = np.sqrt((x - c[0])**2 + (y - c[1])**2)
            if dist < self.distance:
                return False
        return True


class CollisionDetector():
    def __init__(self, a, b):
        def collision_begin(arbiter, space, data):
            self.detected = True
            return True

        def collision_pre_solve(arbiter, space, data):
            return True

        def collision_post_solve(arbiter, space, data):
            return True

        def collision_separate(arbiter, space, data):
            self.detected = False
            return True

        self.collisionHandler = space.add_collision_handler(a, b)#pyMunk CollisionHandler is used
        self.collisionHandler.begin = collision_begin                #to trek collisions
        self.collisionHandler.pre_solve = collision_pre_solve
        self.collisionHandler.post_solve = collision_post_solve
        self.collisionHandler.separate = collision_separate
        self.detected = False


class Obstacle():
    #obstacles are generic bodies of STATIC type(based on pymunk definition)
    #they posses shape, position, and are capable of interacting with DYNAMIC bodies
    def __init__(self, size, location):
        self.size = size
        self.body = pymunk.Body(body_type = pymunk.Body.STATIC)
        self.shape = pymunk.Poly.create_box(self.body, size)
        self.shape.collision_type = 3
        self.shape.color = 25, 230, 179
        self.body.position = location
        self.add_to_space()

    def add_to_space(self):#adding the obstacle to existing pymunk space
        space.add(self.body, self.shape)


class Borders():
    def __init__(self):
        self.t = 5
        self.horiontal_border()
        self.vertical_border()

    def horiontal_border(self):
        upper_border = pymunk.Body(body_type = pymunk.Body.STATIC)
        upper_shape = pymunk.Segment(upper_border, (1,height),
                                     (width-1,height), self.t)
        upper_shape.collision_type = 3
        upper_shape.color = 25, 230, 179

        lower_border = pymunk.Body(body_type = pymunk.Body.STATIC)
        lower_shape = pymunk.Segment(lower_border, (1,1),
                                     (width-1,1), self.t)
        lower_shape.collision_type = 3
        lower_shape.color = 25, 230, 179

        space.add(upper_shape, upper_border, lower_shape, lower_border)

    def vertical_border(self):
        left_border = pymunk.Body(body_type = pymunk.Body.STATIC)
        left_shape = pymunk.Segment(left_border, (0,0),
                                     (0,height), self.t)
        left_shape.collision_type = 3
        left_shape.color = 25, 230, 179

        right_border = pymunk.Body(body_type = pymunk.Body.STATIC)
        right_shape = pymunk.Segment(right_border, (width-1,),
                                     (width-1,height), self.t)
        right_shape.collision_type = 3
        right_shape.color = 25, 230, 179

        space.add(left_shape, left_border, right_shape, right_border)


class Reward():
    def __init__(self, margin):
        self.radius = 10
        self.body = pymunk.Body(1, 1 , body_type = pymunk.Body.DYNAMIC)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.color = red
        self.shape.collision_type = 2

        self.margin = margin
        self.positioning()

        space.add(self.body, self.shape)

    def positioning(self):
        x = randint(self.margin, width - self.margin)
        y = randint(self.margin, height - self.margin)
        allowed = positionHandler.place_reward(x, y)
        if allowed:
            self.body.position = (x, y)
        else:
            self.positioning()


class Agent():
    def __init__(self):
        radius = 15
        moment = pymunk.moment_for_circle(1, 0, radius)
        self.body = pymunk.Body(1, moment)
        self.shape = pymunk.Circle(self.body, radius)
        self.body.position = 50, 50
        self.shape.color = yellow
        self.shape.collision_type = 1
        space.add(self.body, self.shape)

        self.memory_length = 20000
        self.samples_available = 0
        self.memory = []

        self.reward_memory_length = 150
        self.reward_memory = []

        self.exp_min = 0.1
        self.exp_max = 0.9
        self.exploration_coef = self.exp_min + self.exp_max
        self.decay = 0.9975

    def push_to_memory(self, state):
        if len(self.memory) < self.memory_length:
            self.memory.append(state)
        else:
            self.memory[np.random.randint(0,self.memory_length)] = state
        self.samples_available += 1

    def push_to_reward_memory(self, state):
        if len(self.reward_memory) < self.reward_memory_length:
            self.reward_memory.append(state)
        else:
            self.reward_memory[np.random.randint(0,self.reward_memory_length)] = state


class Collector():
    def __init__(self, sonar_arms):
        self.collected = []
        self.sonar_arms = sonar_arms

    def get_position(self):
        for sonar_arm in self.sonar_arms:
            readings = sonar_arm.get_sensors_position()
            self.collected.append(readings)
        position = np.array(self.collected).flatten()
        self.collected = []
        return position

    def get_score(self):
        for sonar_arm in self.sonar_arms:
            readings = sonar_arm.get_sonar_readings()
            self.collected.append(readings)
        score = np.array(self.collected).flatten().sum()
        self.collected = []
        return score


class SonarArm():
    def __init__(self, number_of_points, angle):
        self.distance = 45
        self.angle = np.deg2rad(angle)
        self.step = 20
        self.number_of_points = number_of_points

    def create_sonar_arm(self):
        self.coordinates = []
        i = 0
        for point in range(self.number_of_points):
            r = self.distance + i*self.step
            x = int(drone.body.position[0] + r*np.cos(drone.body.angle+self.angle))
            y = int(drone.body.position[1] + r*np.sin(drone.body.angle+self.angle))
            self.coordinates.append((x, height - y))
            i += 1

    def get_sonar_readings(self):
        readings = []
        for coordinate in self.coordinates:
            if (coordinate[0] < 0 or coordinate[0] > width-1
                or coordinate[1] < 0 or coordinate[1] > height-1 ):
                score = -3
                readings.append(score)
                continue
            reading = [screen.get_at(coordinate)].pop()
            score = 0
            if reading == blue:
                score = -2
            readings.append(score)
        return np.array(readings)

    def get_sensors_position(self):
        position = []
        i = 1
        for coordinate in self.coordinates:
            if (coordinate[0] < 0 or coordinate[0] > width-1
                or coordinate[1] < 0 or coordinate[1] > height-1 ):
                position.append(10./i)
                i += 1
                continue
            reading = [screen.get_at(coordinate)].pop()
            pos = 0
            if reading == blue:
                pos = 10./i
            position.append(pos)
            i += 1
        return np.array(position)


class NeuralNetwork():
    def __init__(self, host):
        self.host = host
        self.gamma = 0.9
        self.input = 28
        self.output = 3
        self.model = self.make_model()
        self.target_model = self.make_model()

    def make_model(self):
        model = Sequential()

        model.add(Dense(128, input_dim = self.input,
                        activation = 'relu'))
        model.add(Dense(128, activation = 'linear'))
        model.add(Dense(self.output))

        model.compile(loss='mse',
                      optimizer=RMSprop(learning_rate=0.001))

        return model

    def learn_from_sample(self, sample_size):
        batch = random.sample(self.host.memory, sample_size)
        for position, new_position, move, score, done in batch:
            target = self.model.predict(np.array([position]))
            at = move
            if done:
                target[0][at] = score
            else:
                t = self.target_model.predict(np.array([new_position]))
                target[0][at] = score + self.gamma*np.amax(t[0])
            self.model.fit(np.array([position.flatten()]), target, epochs = 1, verbose = 0)

    def update_strategy(self):
        self.host.exp_max *= self.host.decay
        self.host.exploration_coef = self.host.exp_max + self.host.exp_min

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

#=========================================================================#
#=========================================================================#
obstacle1 = Obstacle((160, 160),(width*0.7, height*0.5))
obstacle2 = Obstacle((160, 160),(width*0.3, height*0.5))
obstacle3 = Obstacle((30, 30),(width*0.5, height*0.9))
obstacle4 = Obstacle((30, 30),(width*0.5, height*0.1))
obstacles = [obstacle1, obstacle2, obstacle3, obstacle4]

positionHandler = PositionHandler(obstacles = obstacles,
                                  distance = 100)
borders = Borders()

drone = Agent()

sonar_arm1 = SonarArm(4, 0)
sonar_arm2 = SonarArm(4, 25)
sonar_arm3 = SonarArm(4, 50)
sonar_arm4 = SonarArm(4, 75)
sonar_arm5 = SonarArm(4, -25)
sonar_arm6 = SonarArm(4, -50)
sonar_arm7 = SonarArm(4, -75)
sonar_arms = [sonar_arm1, sonar_arm2, sonar_arm3, sonar_arm4,
              sonar_arm5, sonar_arm6, sonar_arm7]
for sonar_arm in sonar_arms:
    sonar_arm.create_sonar_arm()

collector = Collector(sonar_arms)

ann = NeuralNetwork(host = drone)
velocity = 1

collision_with_obstacle = CollisionDetector(drone.shape.collision_type,
                                          obstacle1.shape.collision_type)

recorder = Score()
#=========================================================================#
#=========================================================================#

def draw_sonars():
    for sonar_arm in sonar_arms:
        for coordinates in sonar_arm.coordinates:
            pygame.draw.circle(screen, (255,255,255), (coordinates), 2)

def take_step():
    space.step(dt)
    screen.fill(black)
    space.debug_draw(options)
    for sonar_arm in sonar_arms:
        sonar_arm.create_sonar_arm()
    #draw_sonars()
    pygame.display.update()

def take_action(same_move, same_action, position):
    decision = np.random.uniform(0, 1)
    if drone.exploration_coef > decision:
        move = np.random.randint(3)
        if move == 0:#keep moving in the same dir
            return same_move, same_action
        if move == 1:#turn left
            drone.body.angle += np.deg2rad(ANGLE)
        if move == 2:#turn right
            drone.body.angle -= np.deg2rad(ANGLE)
        action = Vec2d(velocity, 0).rotated(drone.body.angle)
        return move, action

    else:
        move = np.argmax(ann.model.predict(np.array([position.flatten()]))[0])
        if move == 0:#keep moving in the same dir
            return same_move, same_action
        if move == 1:#turn left
            drone.body.angle += np.deg2rad(ANGLE)
        if move == 2:#turn right
            drone.body.angle -= np.deg2rad(ANGLE)
        action = Vec2d(velocity, 0).rotated(drone.body.angle)
        return move, action

def start_next_episode():
    drone.body.angle = 0
    drone.body.position = 50, 50
    take_step()

run = True
move = 0
action = velocity, 0
score = 1
position = collector.get_position()#shuffling the position to avoid correlation to index
sample_size = 256
t = 1
game = 0
while run and game < 1000:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYUP:
            ann.model.save_weights("model_"+str(game)+".h5")


    new_position =collector.get_position()
    move, action = take_action(move, action, position)
    drone.body.position += action
    take_step()
    if collision_with_obstacle.detected:
        print("Game %d ended" %(game + 1))
        print(str(drone.exploration_coef)+"\n")
        score = -1000
        state = (position, new_position, move, score, True)
        drone.push_to_memory(state)
        move = 0
        action = velocity, 0
        if drone.samples_available >= sample_size:
            ann.learn_from_sample(sample_size)#DRL only 1 ann used
            ann.update_strategy()
        start_next_episode()
        recorder.record(game, t)
        game += 1
        t = 1
        continue
    score = collector.get_score()
    score = 40 + score
    state = (position, new_position, move, score, False)
    drone.push_to_memory(state)
    if t>3500:
        print("Game %d ended. Score limit" %(game + 1))
        print(str(drone.exploration_coef)+"\n")
        move = 0
        action = velocity, 0
        start_next_episode()
        recorder.record(game, t)
        game += 1
        t = 1
        continue
    position = new_position
    t += 1

pygame.quit()
recorder.plot()
