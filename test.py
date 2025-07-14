import carla
import time

client = carla.Client('localhost', 2000)
world = client.get_world()
map = world.get_map()
spectator = world.get_spectator()

time_1 = time.time()

for i in range(100):
    location = spectator.get_transform().location

time_2 = time.time()

print((time_2 - time_1) / 100)