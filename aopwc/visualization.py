import matplotlib.pyplot as plt

def vis_open_loop_tracks(preds, targets, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    px, py = preds[0].detach().cpu().numpy()
    tx, ty = targets[0].detach().cpu().numpy()

    ax.plot(tx, ty, markersize=5, marker='o', label='target')
    ax.plot(px, py, markersize=5, marker='o', label='prediction')

    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel('x-position (arcsecs)')
    ax.set_ylabel('y-position (arcsecs)')

    return ax
