import numpy as np
import matplotlib.pyplot as plt

def plot(c, A, b):
    # adapted https://stackoverflow.com/a/57017638/7696065

    fig, ax = plt.subplots()

    # [xmin, xmax, ymin, ymax]
    ar = [-4, 4, -4, 4] # axis range

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


    plt.axis(ar)

    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    # required for labels
    ax.legend()
    plt.show()
