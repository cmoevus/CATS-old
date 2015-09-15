import os
for mod in os.listdir('.'):
    from mod import *
