import random
import sys

size = 3000000

for x in (random.randint(5,5) for x in range(size)):
    sys.stdout.write(f"{x} ")