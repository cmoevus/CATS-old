#!env /usr/bin/python2
# -*- coding: utf8 -*-
# from cats import extensions, sources, processors, content
from cats import sources, content, processors
import numpy as np
import cats.filters.signal as fsig
from matplotlib import pyplot as plt
from cats.utils.math import gamma_cdf, gamma_pdf
from cats.utils import colors
from mpl_toolkits.mplot3d import Axes3D

s = '/home/corentin/Dropbox/PhD/Data/'
f = s + 'test.ps'

particles = processors.particles.stationary(sources.ROI('/home/corentin/150211. SNAPJPred at21 1per80', t=(0, 10)), 0.5, 0.07)
# particles = content.particles.Particles()
# particles.load(f)
# particles.extend(p)
# particles.max_disp = (2, 2)
# particles.max_blink = 100
# particles.sources[0].t = particles.sources[0].t.start, particles.sources[0].t.start + 2
# particles.process()
particles.abs_position = True
particles.save('test.ps')

print(particles[0][0].item())
print(particles[0][0]['x'])
print(particles[0]['x'])

particles.framerate = 5
print(particles[0][0].framerate)

p = content.particles.Particles()
p.load('test.ps')
print(p.framerate)

# Plot
# c = get_colors(1)[0]
# cdf = particles.blinking_pdf(False)
# plt.plot(cdf[1][:-1], cdf[0], 'o', color=c)
# x = range(0, particles.max_blink)
# # plt.plot(x, gamma_pdf(x, *particles.blinking_pdf(True)[0]), color=c, lw=1.5)
#
# # Cut
# # fsig.blinking_thresholding(particles, 0.01)
#
# print(particles.blinking_pdf()[0])
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n1, n2, n3 = 6, 3, 3
# cs = colors.nuances((255, 45, 0), n1)
# for i in range(n1):
#     cs2 = colors.shades([l * 255 for l in cs[i]], n2, True)
#     for j in range(n2):
#         cs3 = colors.tones([k * 255 for k in cs2[j]], n3, True)
#         for k in range(n3):
#             plt.plot([i + 0.5], [j + 0.5], [k + 0.5], 'o', color=cs3[k], ms=10)
# plt.xlim(xmin=0, xmax=n1)
# plt.ylim(0, n2)
# plt.show()
