import numpy as np

MIN_TIME_DELTA = 0.001

def compute_velocity(pos1, pos2, time_delta):
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(p2 - p1) / time_delta for p1, p2 in zip(pos1, pos2)]


def compute_acceleration(vel1, vel2, time_delta):
    if abs(time_delta) < MIN_TIME_DELTA:
        return [0.0, 0.0]
    return [(v2 - v1) / time_delta for v1, v2 in zip(vel1, vel2)]

def calculate_average_vector(source):
    n = len(source)
    return [sum(v[0] for v in source) / n, sum(v[1] for v in source) / n]

def calculate_magnitude(vector):
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5


def update_lpf(now, value, alpha):
    return alpha * value + (1 - alpha) * now

def avg_magnitude(source):
    return sum(calculate_magnitude(v) for v in source) / len(source)

def calculate_average_point(points) -> list[float]:
    x_sum = sum(p.x for p in points)
    y_sum = sum(p.y for p in points)
    n = len(points)
    return [x_sum / n, y_sum / n]

def scaled_sigmoid(x, k=5.0):
    raw = 1 / (1 + np.exp(-k * (x - 1)))
    return raw / (1 / (1 + np.exp(0)))  # Normalize so f(1) = 1

def asymmetric_sigmoid(x, k1=5.0, k2=0.1):
    if x < 1: return scaled_sigmoid(x, k1)
    else: return scaled_sigmoid(x, k2)
