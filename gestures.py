import math
from itertools import combinations

def vector(a, b):
    return [b.x - a.x, b.y - a.y, b.z - a.z]

def dot(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def is_finger_folded(wrist, mcp, tip):
    base_vec = vector(wrist, mcp)
    tip_vec = vector(mcp, tip)
    return dot(base_vec, tip_vec) < 0

def hand_compactness(hand_landmarks):
    # Use only fingertips (thumb to pinky)
    tip_ids = [4, 8, 12, 16, 20]
    tips = [hand_landmarks.landmark[i] for i in tip_ids]
    pairs = list(combinations(tips, 2))
    distances = [distance(a, b) for a, b in pairs]
    return sum(distances) / len(distances)  # Average distance

def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    finger_indices = [(5, 8), (9, 12), (13, 16), (17, 20)]  # MCP, TIP
    folded_fingers = sum(
        is_finger_folded(wrist, hand_landmarks.landmark[mcp], hand_landmarks.landmark[tip])
        for mcp, tip in finger_indices
    )

    compactness = hand_compactness(hand_landmarks)

    # Heuristic: compactness below threshold and fingers folded
    return folded_fingers >= 3 and compactness < 0.09