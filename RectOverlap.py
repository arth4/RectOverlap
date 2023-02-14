import numpy as np
from collections import namedtuple

Line = namedtuple('Line', ['center', 'direction'])
Rect = namedtuple('Rect', ['center', 'size', 'angle'])


def is_overlapping(rect0_center, rect0_size, rect0_angle, rect1_center, rect1_size, rect1_angle, radians=True):
    if not radians:
        rect0_angle = np.radians(rect0_angle)
        rect1_angle = np.radians(rect1_angle)
    rect0 = Rect(rect0_center, rect0_size, rect0_angle)
    rect1 = Rect(rect1_center, rect1_size, rect1_angle)
    return is_overlapping_rect(rect0, rect1)

def is_overlapping_rect(rect0, rect1):
    return is_projection_colliding(rect0, rect1) and is_projection_colliding(rect1, rect0)

def is_projection_colliding(rect, other_rect):
    axes = get_axes(other_rect)
    corners = get_corners(rect)
    signed_dists = np.zeros(len(corners))
    for i, axis in enumerate(axes):
        rect_half_size = other_rect.size[i] / 2
        for j, corner in enumerate(corners):
            signed_dists[j] = signed_distance(other_rect, axis, corner)
        mn, mx = np.min(signed_dists), np.max(signed_dists)

        if (not (mn < 0 and mx > 0  or np.abs(mn) < rect_half_size or np.abs(mx) < rect_half_size)):
            return False
    return True

def get_axes(rect):
    RX = rotate_vector(np.array([1, 0]), rect.angle)
    RY = rotate_vector(np.array([0, 1]), rect.angle)
    line0 = Line(rect.center, RX)
    line1 = Line(rect.center, RY)
    return line0, line1

def get_corners(rect):
    axis0, axis1 = get_axes(rect)
    RX = axis0.direction * rect.size[0] / 2
    RY = axis1.direction * rect.size[1] / 2
    return np.array([rect.center + RX + RY, rect.center + RX - RY, rect.center - RX - RY, rect.center - RX + RY])

def project(vector, line):
    return line.center + line.direction * np.dot(vector - line.center, line.direction)

def rotate_vector(point, angle):
    return np.dot(point, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))

def magnitude(vector):
    return np.linalg.norm(vector)

def signed_distance(rect, line, corner):
    projected = project(corner, line)
    CP = projected - rect.center
    sign = 1 if np.dot(CP, line.direction) > 0 else -1
    return magnitude(CP) * sign

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def random_rect():
        return np.random.uniform(-1, 1, 2), np.random.uniform(0.5, 1, 2), np.random.uniform(0, 360)

    def draw_axes(rect, color='black'):
        axes = get_axes(rect)
        for axis in axes:
            point0 = axis.center - axis.direction * 2
            point1 = axis.center + axis.direction * 2
            plt.plot([point0[0], point1[0]], [point0[1], point1[1]], linestyle='--', color=color, alpha=0.5)

    def draw_corner_projections(rect, other_rect, color='black'):
        axes = get_axes(other_rect)
        corners = get_corners(rect)
        for axis in axes:
            for corner in corners:
                projected = project(corner, axis)
                plt.scatter(projected[0], projected[1], color=color, alpha=0.7, marker='x')
    fig, axs = plt.subplots(2, 3)
    for ax in axs.flatten():
        plt.sca(ax)
        rect0_center, rect0_size, rect0_angle = random_rect()
        rect1_center, rect1_size, rect1_angle = random_rect()

        ax.set_aspect('equal')
        ax.add_patch(Rectangle(rect0_center - rect0_size / 2, *rect0_size, angle=360-rect0_angle, fill=False, rotation_point="center", color="red"))
        ax.add_patch(Rectangle(rect1_center - rect1_size / 2, *rect1_size, angle=360-rect1_angle, fill=False, rotation_point="center", color="blue"))
        ax.autoscale_view()
        rect0 = Rect(rect0_center, rect0_size, np.radians(rect0_angle))
        rect1 = Rect(rect1_center, rect1_size, np.radians(rect1_angle))
        
        overlap = is_overlapping_rect(rect0, rect1)
        draw_axes(rect0, color='red')
        draw_axes(rect1, color='blue')

        draw_corner_projections(rect0, rect1, color='red')
        draw_corner_projections(rect1, rect0, color='blue')


        plt.title('Overlapping' if overlap else 'Not Overlapping')
    plt.show()
