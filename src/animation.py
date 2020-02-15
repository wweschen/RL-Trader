"""
==================
Animated histogram
==================

Use a path patch to draw a bunch of rectangles for an animated histogram.
"""

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

# Fixing random state for reproducibility
class TestAni():
    def __init__(self):

        np.random.seed(19680801)

        # histogram our data with numpy
        self.data = np.random.randn(1000)
        self.n, self.bins = np.histogram(self.data, 100)

        # get the corners of the rectangles for the histogram
        self.left = np.array(self.bins[:-1])
        self.right = np.array(self.bins[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.n
        nrects = len(self.left)

        ###############################################################################
        # Here comes the tricky part -- we have to set up the vertex and path codes
        # arrays using ``plt.Path.MOVETO``, ``plt.Path.LINETO`` and
        # ``plt.Path.CLOSEPOLY`` for each rect.
        #
        # * We need 1 ``MOVETO`` per rectangle, which sets the initial point.
        # * We need 3 ``LINETO``'s, which tell Matplotlib to draw lines from
        #   vertex 1 to vertex 2, v2 to v3, and v3 to v4.
        # * We then need one ``CLOSEPOLY`` which tells Matplotlib to draw a line from
        #   the v4 to our initial vertex (the ``MOVETO`` vertex), in order to close the
        #   polygon.
        #
        # .. note::
        #
        #   The vertex for ``CLOSEPOLY`` is ignored, but we still need a placeholder
        #   in the ``verts`` array to keep the codes aligned with the vertices.
        nverts = nrects * (1 + 3 + 1)
        self.verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

###############################################################################
# To animate the histogram, we need an ``animate`` function, which generates
# a random set of numbers and updates the locations of the vertices for the
# histogram (in this case, only the heights of each rectangle). ``patch`` will
# eventually be a ``Patch`` object.
        self.patch = None




###############################################################################
# And now we build the `Path` and `Patch` instances for the histogram using
# our vertices and codes. We add the patch to the `Axes` instance, and setup
# the `FuncAnimation` with our animate function.
        self.fig, ax = plt.subplots()
        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(
            barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
        ax.add_patch(self.patch)

        ax.set_xlim(self.left[0], self.right[-1])
        ax.set_ylim(self.bottom.min(), self.top.max())

    def animate(self, i):
        # simulate new data coming in
        self.data = np.random.randn(1000)
        self.n, self.bins = np.histogram(self.data, 100)
        top = self.bottom + self.n
        self.verts[1::5, 1] = top
        self.verts[2::5, 1] = top
        return [self.patch]

def main():

    t=TestAni()
    ani = animation.FuncAnimation(t.fig, t.animate, interval=100,  blit=True)
    plt.show()


if __name__ == '__main__':
  main()
