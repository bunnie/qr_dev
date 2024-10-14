import cv2
import numpy as np

from ref_images import *

def histogram(data):
    buckets = 256 * [0]
    for d in data.flat:
        buckets[d] += 1
    return buckets

# fragment is byte-wide data that has a value of either 0 or 255
# returns a tuple of (run color, run length, remaining fragment data)
def color_run(fragment):
    run_color = fragment[0] # current run color is by definition the first color of the fragment
    run_length = 1
    for f in fragment:
        if f == run_color:
            run_length += 1
        else:
            break
    return (run_color, run_length, fragment[run_length:])


# search a line of data for a 1:1:3:1:1 ratio of black:white:black:white:black
# this uses the "state machine" method
# "row_normal" means we're searching by rows
def finder_finder_sm(y, line, row_normal=True):
    # method:
    #  - core primitive is advance-to-next-color. This returns a run length and
    #    color argument
    x = 0
    sequence = [(0,0,0)] * 5
    candidates = []
    while len(line) > 0 :
        (color, run_length, line) = color_run(line)
        x += run_length
        sequence = sequence[1:] + [(run_length, color, x)]
        if sequence[0][0] != 0 and sequence[0][1] == 0: # sequence of 5 and black is in the right position
            # print(sequence)
            ratios = []
            denom = sequence[0][0]
            for seq in sequence:
                ratios.append(seq[0] / denom)
            LOWER_1 = 0.5
            UPPER_1 = 2.0
            LOWER_3 = 2.0
            UPPER_3 = 4.0
            if ratios[1] >= LOWER_1 and ratios[1] <= UPPER_1 and ratios[2] >= LOWER_3 and ratios[2] <= UPPER_3 \
                and ratios[3] >= LOWER_1 and ratios[3] <= UPPER_1 and ratios[4] >= LOWER_1 and ratios[4] <= UPPER_1:
                if row_normal:
                    # print(f"{sequence[2][2]},{y} -- {ratios}")
                    candidates.append((sequence[2][2] - sequence[2][0] // 2 - 1, y))
                else:
                    # print(f"{y}, {sequence[2][2]} -- {ratios}")
                    candidates.append((y, sequence[2][2] - sequence[2][0] // 2 - 1))
    return candidates


if __name__ == "__main__":
    qr_raw = cv2.imread('images/test5.png')

    cv2.imshow('image', qr_raw)

    # binarization
    # locate finder patterns
    # affine transform
    # version number extract
    # find alignment patterns
    # regional alignment
    # create bitstream
    # ECC
    # extract data

    # binarization: histogram, find bimodal peaks, slice in the middle
    if False:
        # histogram, find bimodal peaks, slice in the middle
        buckets = histogram(qr_raw)
        i_max1 = 0
        v_max1 = 0
        i_max2 = 0
        v_max2 = 0
        for (i, bucket) in enumerate(buckets):
            if bucket > v_max1:
                v_max2 = v_max1
                i_max2 = i_max1
                v_max1 = bucket
                i_max1 = i

        print("foo")
    else:
        # average, slice at half of average
        mean = np.mean(qr_raw)
        print(f"slicing at {mean}")
        np.zeros(qr_raw.size, np.uint8)
        (_r, rgb) = cv2.threshold(qr_raw, mean, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cv2.imshow('binary', binary)

    # locate finder patterns, based on a 1:1:3:1:1 pattern
    row_candidates = []
    for i, row in enumerate(binary):
        candidate = finder_finder_sm(i, row, row_normal=True)
        if len(candidate) > 0:
            row_candidates += candidate

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in row_candidates:
        cv2.line(overlay, candidate, candidate, (0, 255, 0), 1)
    blended = cv2.addWeighted(overlay, 1.0, rgb, 0.5, 0)

    print("columns")
    col_candidates = []
    for i, col in enumerate(binary.T):
        candidate = finder_finder_sm(i, col, row_normal=False)
        if len(candidate) > 0:
            col_candidates += candidate

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in col_candidates:
        cv2.line(overlay, candidate, candidate, (255, 0, 255), 1)

    blended = cv2.addWeighted(overlay, 1.0, blended, 0.5, 0)

    intersected = np.zeros(rgb.shape, np.uint8)
    marks = []
    for candidate in list(set(row_candidates).intersection(set(col_candidates))):
        cv2.line(intersected, candidate, candidate, (0, 255, 0), 1)
        print(f"{candidate}")
        marks += [candidate]
    blended_intersection = cv2.addWeighted(intersected, 1.0, rgb, 0.5, 0)

    cv2.imshow('blended', blended)
    cv2.imshow('intersection', blended_intersection)
    cv2.waitKey(0)

    if len(marks) == 3:
        print("candidate good")
