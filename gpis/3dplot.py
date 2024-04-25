import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.ticker import LinearLocator
from argparse import ArgumentParser

from gpis import topcd

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="gpis")
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--axis", type=str, default="z")
parser.add_argument("--isf_limit", type=float, default=0.2)
parser.add_argument("--quiver_spacing", type=int, default=5)
parser.add_argument("--query_point", type=float, nargs=3, default=None)

args = parser.parse_args()

data = np.load(args.data, allow_pickle=True)["data"].item() 
test_mean = data["mean"]
test_var = data["var"]
test_normal = data["normal"]
num_steps = test_mean.shape[0]
bound = data["bound"]
center = data["center"]

vis_points, vis_normals, vis_var = topcd(
    test_mean,
    test_normal,
    [-bound+center[0],-bound+center[1],-bound+center[2]],
    [bound+center[0],bound+center[1],bound+center[2]],
    test_var=test_var,
    steps=num_steps,
)

# Set cross section plane to visualize
if args.axis == "x": 
    # y-z plane
    X, Y = np.meshgrid(np.linspace(data["lb"][1],data["ub"][1],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
    x_label = "y"
    y_label = "z"
    level_coords = np.meshgrid(
        np.linspace(data["lb"][0],data["ub"][0],num_steps), indexing="xy"
    )[0]
elif args.axis=="y":
    # x-z plane
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy")
    x_label = "x"
    y_label = "z"
    level_coords = np.meshgrid(
        np.linspace(data["lb"][1],data["ub"][1],num_steps), indexing="xy"
    )[0]
else:
    # x-y plane
    X, Y = np.meshgrid(np.linspace(data["lb"][0],data["ub"][0],num_steps),
                    np.linspace(data["lb"][1],data["ub"][1],num_steps), indexing="xy")
    x_label = "x"
    y_label = "y"
    level_coords = np.meshgrid(
        np.linspace(data["lb"][2],data["ub"][2],num_steps), indexing="xy"
    )[0]

if args.query_point is not None:
    print(args.query_point)
    if args.axis == "x":
        q_point = np.array([args.query_point[1], args.query_point[2], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][0],data["ub"][0],num_steps)-args.query_point[0]))
    elif args.axis == "y":
        q_point = np.array([args.query_point[0], args.query_point[2], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][1],data["ub"][1],num_steps)-args.query_point[1]))
    else:
        q_point = np.array([args.query_point[0], args.query_point[1], 0.0])
        Z_index = np.argmin(np.abs(np.linspace(data["lb"][2],data["ub"][2],num_steps)-args.query_point[2]))
    
for i in range(0, num_steps, args.stride):
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(f"Cross section axis = {args.axis} | level = {i} | {args.axis}-coord = {level_coords[i]}")
    
    ax = fig.add_subplot(221, projection='3d', computed_zorder=False)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    if args.query_point is not None:
        i = Z_index

    # Z value to be mean
    # Set color to be log(variance)
    if args.axis == "x":
        Z = test_mean[i]
        color_dimension = np.log(test_var[i])
        normal_vec = test_normal[i]
    elif args.axis == "y":
        Z = test_mean[:,i]
        color_dimension = np.log(test_var[:,i])
        normal_vec = test_normal[:,i]
    else:
        Z = test_mean[:,:,i]
        color_dimension = np.log(test_var[:,:,i])
        normal_vec = test_normal[:,:,i]


    # Plot 1: Mean value as z, color indicates variance
    # Choose colormap and scale to min and max of log(variance)
    ax.set_title("mean")
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, minn + (maxx-minn)/2)
    m = cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    surf = ax.plot_surface(X, Y, Z,facecolors=fcolors,
                        linewidth=0, antialiased=True, alpha=1)#, zorder=2)
    if args.query_point is not None:
        ax.scatter([q_point[0]],[q_point[1]], [q_point[2]], color="orange", s=100, zorder=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("mean", rotation=90)
    # Customize the z axis.
    ax.set_zlim(-args.isf_limit, args.isf_limit)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps log(var) values to colors.
    plt.colorbar(m, ax=ax, shrink=0.5, aspect=5, label="log(var)")
    # Add plane at z=0
    ax.plot_surface(X, Y, np.zeros_like(Y), alpha=0.5)

    # Plot 2: Normal field and contour
    ax2.set_title("normal field and contour")
    normal_field = ax2.quiver(X[::args.quiver_spacing,::args.quiver_spacing],
                              Y[::args.quiver_spacing,::args.quiver_spacing], 
                              np.zeros_like(Z)[::args.quiver_spacing,::args.quiver_spacing], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,0], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,1], 
                              normal_vec[::args.quiver_spacing,::args.quiver_spacing,2], length=0.005, normalize=True)
    ax2.contour(X, Y, Z, levels = np.linspace(-0.001, 0.001, 2),cmap="jet", alpha=0.5)
    if args.query_point is not None:
        ax2.scatter([q_point[0]],[q_point[1]], [q_point[2]], color="orange", s=100, zorder=10)
    
    # Plot 3: input point cloud
    ax3.set_title("input and predicted point clouds")
    ax3.axis('equal')
    vis_var = vis_var / vis_var.max()
    rgb = np.zeros_like(vis_points)
    rgb[:,0] = vis_var
    rgb[:,2] = 1 - vis_var
    alpha = np.ones((rgb.shape[0], 1)) * 0.05
    colors = np.concatenate((rgb, alpha), axis=1)
    ax3.scatter(
        vis_points[:, 0],
        vis_points[:, 1],
        vis_points[:, 2],
        c = colors,
        marker="o",
    )
    gt_rgb = np.zeros_like(data["points"])
    gt_rgb[:,1] = 1
    gt_alpha = np.ones((gt_rgb.shape[0], 1))
    gt_colors = np.concatenate((gt_rgb, gt_alpha), axis=1)
    ax3.scatter(
        data["points"][:, 0],
        data["points"][:, 1],
        data["points"][:, 2],
        c=gt_colors,
        marker=".",
    )
    # Set axes bounds and labels
    ax3.set(
        xlim=(data["lb"][0], data["ub"][0]),
        ylim=(data["lb"][1], data["ub"][1]),
        zlim=(data["lb"][2], data["ub"][2]),
    )
    green_patch = mpatches.Patch(color=[0,1,0], label="Ground truth")
    red_patch = mpatches.Patch(color=[1,0,0], label="Predicted - higher var")
    blue_patch = mpatches.Patch(color=[0,0,1], label="Predicted - lower var")
    ax3.legend(handles=[red_patch, blue_patch, green_patch], loc="upper center", ncols=2)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    # Add cross section plane
    if args.axis == "x":
        ax3.plot_surface(level_coords[i]**np.ones_like(Y), X, Y, alpha=0.5)
    elif args.axis == "y":
        ax3.plot_surface(X, level_coords[i]**np.ones_like(Y), Y, alpha=0.5)
    else:
        ax3.plot_surface(X, Y, level_coords[i]**np.ones_like(Y), alpha=0.5)
    
    # Plot 4: 2D cross section plot where colors are mean values
    # Set color bound to be zero center (white = 0)
    ax4.set_title("mean")
    cm_bound = min(np.abs(np.max(Z)), np.abs(np.abs(np.min(Z))))
    mean_cm_norm = matplotlib.colors.Normalize(-cm_bound, cm_bound)
    if args.axis == "x":
        im = ax4.imshow(test_mean[i], cmap="seismic", norm=mean_cm_norm)
    elif args.axis == "y":
        im = ax4.imshow(test_mean[:,i], cmap="seismic", norm=mean_cm_norm)
    else:
        im = ax4.imshow(test_mean[:,:,i], cmap="seismic", norm=mean_cm_norm)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel(y_label)
    plt.colorbar(im, ax=ax4, shrink=0.5, aspect=5, label="mean")

    plt.show()

    if args.query_point is not None:
        break
