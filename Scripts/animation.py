import matplotlib.animation as animation
from glob import glob
from Utils import patient
import sys



root = sys.argv[1]
patient_no = sys.argv[2]







fig = plt.figure()
anim = plt.imshow(


def animate(pat, gifname):
    fig = plt.figure()
    anim = plt.imshow(pat[50])
    def update(i):
        anim.set_array(pat[i])
        return anim,
    
    a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=50, blit=True)
    a.save(gifname, writer='imagemagick')
    
