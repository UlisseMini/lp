import numpy as np
import matplotlib.pyplot as plt

# [xmin, xmax, ymin, ymax]
ar = [-4, 4, -4, 4] # axis range

def plot3d(c, A, b, path, num=100):
    ax = plt.subplot(122, projection='3d')

    # Draw path
    # plot_path3d(ax, path, c=c)
    cp = [10*c@x for x in path]
    print('cp', cp)
    ax.plot([x[0] for x in path], [x[1] for x in path], cp, color='red')

    # Draw cost function
    xs = np.linspace(ar[0], ar[1], num)
    ys = np.linspace(ar[2], ar[3], num)
    X, Y = np.meshgrid(xs, ys)
    Z = c[0]*X + c[1]*Y
    ax.plot_surface(X, Y, Z)

    # Draw constraint lines

    # ax.arrow(*x, *step, width=0.05)


def plot(c, A, b, path):
    # adapted https://stackoverflow.com/a/57017638/7696065
    # TODO: Plot cost colors, plot optimal point computed by scipy

    # 1 row, 2 cols, id 1, plot3d uses id 2.
    ax = plt.subplot(121)

    # barrier lines
    for i in range(len(b)):
        standard_form   = f'{A[i][0]}x + {A[i][1]}y = {b[i]}'
        # slope_intercept = f'y = {-A[i][0]/A[i][1]:.3f}x + {b[i]/A[i][1]:.3f}'

        x = np.linspace(ar[0], ar[1], num=2)
        y = (b[i] - A[i][0]*x) / A[i][1]

        ax.plot(x, y, label=f'${standard_form}$')


    # feasible reigion
    n = 300
    x, y = np.meshgrid(np.linspace(ar[0], ar[1], n), np.linspace(ar[2], ar[3], n))
    feasible = (x > 0) & (y > 0)
    for i in range(len(b)):
        print(f'{A[i][0]}x + {A[i][1]}y <= {b[i]}')
        feasible &= (A[i][0]*x + A[i][1]*y <= b[i])

    ax.imshow(
        feasible,
        extent=(x.min(),x.max(),y.min(),y.max()),
        origin='lower', cmap='Greys', alpha=0.3
    )

    # path
    plot_path(ax, path, c=c)

    # Gradient
    plt.scatter([path[-1][0]], [path[-1][1]])
    plt.arrow(*path[-2], *(path[-1]-path[-2]), width=0.05)

    # Axis and gridlines
    plt.axis(ar)

    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    # required for labels
    ax.legend()



def plot_path3d(ax, path, c, color='red'):
    ax.plot([x[0] for x in path], [x[1] for x in path], [c@x for x in path], color=color)

def plot_path(ax, path, c, color='red'):
    ax.plot([x[0] for x in path], [x[1] for x in path], color=color)
