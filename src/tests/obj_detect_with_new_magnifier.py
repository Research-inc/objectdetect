import sys
sys.path.append("../")
from core import detectWithSearchName_v2, magnifier, closenessHelper

quad, init_width, init_height = detectWithSearchName_v2('test3.jpeg', "bottle")

quad, new_width, new_height = detectWithSearchName_v2('test4.jpeg', "bottle")

closenessHelper(magnifier(init_height, new_height), quad)