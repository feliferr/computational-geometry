import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import math
from scipy import spatial
from itertools import chain

def pares_circulares(iterable):
    iterable = iter(iterable)
    first = last = next(iterable)
    for x in iterable:
        yield last, x
        last = x
    yield (last, first)

def near_zero(v):
    if isinstance(v, (float, int)):
        return v > -1E-6 and v < 1E-6
    else:
        return np.allclose(v, np.zeros(np.shape(v)))

def calcula_normal(polygon):
    sum = 0
    for (x1, y1), (x2, y2) in pares_circulares(polygon):
        sum += (x2 - x1) * (y2 + y1)
    if sum > 1E-6:
        return 1
    elif sum < -1E-6:
        return -1
    else:
        raise ValueError("Nenhuma normal encontrada")

def fatias_circulares(seq, start, count):
    l = len(seq)
    for i in range(start, start + count):
        yield seq[i % l]

def fatias_circulares_inv(seq, start, count):
    if start + count > len(seq):
        return seq[start + count - len(seq): start]
    else:
        return chain(seq[:start], seq[start + count:])

def existe_pontos_no_triangulo(triangle, points):
    a, b, c = triangle
    s = b - a
    t = c - a
    stack = [s, t]
    if len(s) == 3:
        stack.append(np.cross(s, t))
    mtrx = np.linalg.inv(np.vstack(stack).transpose())
    if len(s) == 3:
        mtrx = mtrx[:2]
    for point in points:
        ps, pt = np.dot(mtrx, point - a)
        if ps >= 0 and pt >= 0 and ps + pt <= 1:
            return True
    return False

def triangulacao(polygon):

    polygon = [np.array(x) for x in polygon]
    normal = calcula_normal(polygon)
    i = 0
    while len(polygon) > 2:
        (a, b, c) = fatias_circulares(polygon, i, 3)
        triangle = (a, b, c)
        if ((a == b).all() or (b == c).all()):
            # Pulando vertices duplicados
            del polygon[(i + 1) % len(polygon)]
            continue

        x = np.cross(c - b, b - a)
        dot = np.dot(normal, x)
        yld = False
        if dot > 1E-6:
            triangle = (a, b, c)
            if not existe_pontos_no_triangulo(triangle,
                                         fatias_circulares_inv(polygon, i, 3)):
                del polygon[(i + 1) % len(polygon)]
                yield triangle
                i = 0
                yld = True
        if not yld:
            i += 1

def plot_triangulation(df, x_list, y_list):
    xs = df['x'].tolist()
    ys = df['y'].tolist()

    xs.append(df['x'].values[0])
    ys.append(df['y'].values[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x_list,y_list)
    plt.plot(xs,ys)
    plt.scatter(xs, ys)
    for xy in zip(xs, ys):                                       # <--
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') #
    plt.show()

def getPolygonPoints(df):
	xs = df['x'].tolist()
	ys = df['y'].tolist()
	lista = []
	for i in range(len(xs)):
		lista.append((xs[i],ys[i]))
	return lista

def make_triangulation(df):
    poly = getPolygonPoints(df)
    print("--> polygon: {}".format(poly))
    tris = list(triangulacao(poly))
    print("--> triangulation: {}".format(tris))

    x_list = []
    y_list = []
    for triangle in tris:
        x_list.append(triangle[0][0])
        y_list.append(triangle[0][1])

        x_list.append(triangle[1][0])
        y_list.append(triangle[1][1])

        x_list.append(triangle[2][0])
        y_list.append(triangle[2][1])

    plot_triangulation(df, x_list, y_list)


df_polygon1 = pd.read_csv("data/polygon1.txt", sep="   ", header=None, names=["x","y"])
df_polygon2 = pd.read_csv("data/polygon2.txt", sep="   ", header=None, names=["x","y"])

make_triangulation(df_polygon1)
make_triangulation(df_polygon2)


