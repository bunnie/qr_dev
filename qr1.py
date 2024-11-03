import cv2
import numpy as np

from ref_images import *
from math import atan2, cos, sin
from homography import *

FIXED_POINT_SHIFT = 16
FIXED_POINT_ONE = 1 << FIXED_POINT_SHIFT

def draw_cross(image, point, color):
    cv2.line(image, (point[0] - 2, point[1]), (point[0] + 2, point[1]), color, 1)
    cv2.line(image, (point[0], point[1] - 2), (point[0], point[1] + 2), color, 1)

def pt_from_hv_line(hline, vline):
    (m1, b1) = hline
    (m2v, b2v) = vline
    m2 = 1 / m2v
    b2 = - b2v / m2v
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (int(x), int(y))

class SearchDirection:
    def __init__(self, point, shape):
        self.point = point
        if point[0] < shape[1] // 2 and point[1] < shape[0] // 2:
            self.direction = "sw"
        if point[0] > shape[1] // 2 and point[1] < shape[0] // 2:
            self.direction = "se"
        if point[0] < shape[1] // 2 and point[1] > shape[0] // 2:
            self.direction = "nw"
        if point[0] > shape[1] // 2 and point[1] > shape[0] // 2:
            self.direction = "ne"
        self.shape = shape

    def as_int(self):
        if self.direction == "sw":
            return (-1, -1)
        elif self.direction == "se":
            return (1, -1)
        elif self.direction == "nw":
            return (-1, 1)
        elif self.direction == "ne":
            return (1, 1)

    def center_point(self):
        return self.point

    def inline_with(self, query):
        # self is the missing point
        if self.direction[0] in query.direction:
            return "h"
        elif self.direction[1] in query.direction:
            return "v"
        else:
            return None

    def dest_point(self, margin = 4):
        if self.direction == "nw":
            return (margin, self.shape[0] - margin)
        elif self.direction == "ne":
            return (self.shape[1] - margin, self.shape[0] - margin)
        elif self.direction == "sw":
            return (margin, margin)
        elif self.direction == "se":
            return (self.shape[1] - margin, margin)
        return None

def best_fit_line_with_outlier_rejection(points, threshold=1.0, max_iterations=10):
    # Step 1: Initial least-squares fit
    def compute_least_squares(points):
        n = len(points)
        sum_x = sum(point[0] for point in points)
        sum_y = sum(point[1] for point in points)
        sum_xx = sum(point[0] ** 2 for point in points)
        sum_xy = sum(point[0] * point[1] for point in points)

        # Slope (m) and intercept (b) of y = mx + b
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        b = (sum_y - m * sum_x) / n
        return m, b

    # Step 2: Calculate residuals (distances from line)
    def calculate_residuals(points, m, b):
        residuals = []
        for x, y in points:
            predicted_y = m * x + b
            residual = abs(y - predicted_y)
            residuals.append((x, y, residual))
        return residuals

    # Step 3: Filter out points with high residuals
    def filter_points(points, residuals, threshold):
        return [(x, y) for (x, y, residual) in residuals if residual <= threshold]

    # Initial fit and outlier rejection loop
    for _ in range(max_iterations):
        m, b = compute_least_squares(points)
        residuals = calculate_residuals(points, m, b)

        # Calculate threshold dynamically if desired
        # E.g., 1.5 times the median residual can be used as threshold
        residual_values = [residual for _, _, residual in residuals]
        median_residual = sorted(residual_values)[len(residual_values) // 2]
        threshold = max(threshold, 1.5 * median_residual)

        # Filter points and repeat fit
        filtered_points = filter_points(points, residuals, threshold)

        # Stop if no points are filtered out
        if len(filtered_points) == len(points):
            break

        points = filtered_points

    # Final fit with filtered points
    m, b = compute_least_squares(points)
    return m, b

class AffineTransform:
    def __init__(self, a, b, c, d, tx, ty):
        self.a = self.float_to_fixed(a)
        self.b = self.float_to_fixed(b)
        self.c = self.float_to_fixed(c)
        self.d = self.float_to_fixed(d)
        self.tx = self.float_to_fixed(tx)
        self.ty = self.float_to_fixed(ty)

    def float_to_fixed(self, value):
        return int(value * FIXED_POINT_ONE)

def get_pixel(image, width, height, x, y):
    # Check for out-of-bounds coordinates, return 0 if outside
    if x < 0 or y < 0 or x >= width or y >= height:
        return 0  # Black for out-of-bounds pixels
    return image[y, x]

def affine_transform(src, tform):
    height, width = src.shape
    dst = np.zeros_like(src, dtype=np.uint8)  # Create an empty output image

    for y_dst in range(height):
        for x_dst in range(width):
            # Inverse affine transformation to get the source pixel coordinates
            x_src_fixed = (tform.a * x_dst + tform.b * y_dst + tform.tx) >> FIXED_POINT_SHIFT
            y_src_fixed = (tform.c * x_dst + tform.d * y_dst + tform.ty) >> FIXED_POINT_SHIFT

            # Set the destination pixel using nearest-neighbor interpolation
            dst[y_dst, x_dst] = get_pixel(src, width, height, int(x_src_fixed), int(y_src_fixed))

    return dst


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
    avg_width = 0
    while len(line) > 0 :
        (color, run_length, line) = color_run(line)
        x += run_length
        sequence = sequence[1:] + [(run_length, color, x)]
        if sequence[0][0] != 0 and sequence[0][1] == 0: # sequence of 5 and black is in the right position
            # print(sequence)
            ratios = []
            denom = sequence[0][0]
            for seq in sequence:
                ratios.append(int(seq[0] / denom))
            LOWER_1 = 0 # 0.5 ideally, but have to go to 0 for fixed point impl
            UPPER_1 = 2
            LOWER_3 = 2
            UPPER_3 = 4
            if ratios[1] >= LOWER_1 and ratios[1] <= UPPER_1 and ratios[2] >= LOWER_3 and ratios[2] <= UPPER_3 \
                and ratios[3] >= LOWER_1 and ratios[3] <= UPPER_1 and ratios[4] >= LOWER_1 and ratios[4] <= UPPER_1:
                if row_normal:
                    print(f"{sequence[2][2]},{y} -- {ratios}")
                    candidates.append((sequence[2][2] - sequence[2][0] // 2 - 1, y))
                else:
                    print(f"{y}, {sequence[2][2]} -- {ratios}")
                    candidates.append((y, sequence[2][2] - sequence[2][0] // 2 - 1))
                for s in sequence:
                    avg_width += s[0]

    return (candidates, int(avg_width / max(len(candidates), 1)))

BW_THRESH = 128

if __name__ == "__main__":
    qr_raw = cv2.imread('images/test256b.png')

    if False:
        cv2.imshow('image', cv2.flip(qr_raw, 0))

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
        # mean = np.mean(qr_raw)
        # print(f"slicing at {mean}")
        # np.zeros(qr_raw.size, np.uint8)
        (_r, rgb) = cv2.threshold(qr_raw, BW_THRESH, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if False:
            cv2.imshow('binary', cv2.flip(binary, 0))

    # locate finder patterns, based on a 1:1:3:1:1 pattern
    row_candidates = []
    row_widths = []
    for i, row in enumerate(binary):
        (candidate, width) = finder_finder_sm(i, row, row_normal=True)
        if len(candidate) > 0:
            row_candidates += candidate
            row_widths += [(candidate[0], width)]

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in row_candidates:
        cv2.line(overlay, candidate, candidate, (0, 255, 0), 1)
    blended = cv2.addWeighted(overlay, 1.0, rgb, 0.5, 0)

    print("columns")
    col_candidates = []
    for i, col in enumerate(binary.T):
        (candidate, _width) = finder_finder_sm(i, col, row_normal=False)
        if len(candidate) > 0:
            col_candidates += candidate

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in col_candidates:
        cv2.line(overlay, candidate, candidate, (255, 0, 255), 1)

    blended = cv2.addWeighted(overlay, 1.0, blended, 0.5, 0)

    intersected = np.zeros(rgb.shape, np.uint8)
    marks = []
    finder_width = 0
    for candidate in list(set(row_candidates).intersection(set(col_candidates))):
        cv2.line(intersected, candidate, candidate, (0, 255, 0), 1)
        for r in row_widths:
            if candidate == r[0]:
                finder_width += r[1]
        print(f"{candidate}")
        marks += [candidate]

    if len(marks) == 3:
        print("candidate good")
    else:
        exit(0)

    finder_width = int(finder_width / len(marks))
    finder_width += (max(int(finder_width / (1+1+3+1+1)), 1) + 2)
    estimated_point = (marks[0][0] + marks[2][0] - marks[1][0], marks[0][1] + marks[2][1] - marks[1][1])
    draw_cross(intersected, estimated_point, (255, 255, 0))

    blended_intersection = cv2.addWeighted(intersected, 1.0, rgb, 0.5, 0)
    if False:
        cv2.imshow('blended', cv2.flip(blended, 0))
        cv2.imshow('intersection', cv2.flip(blended_intersection, 0))
        # cv2.waitKey(0)

    # marks += [estimated_point]
    directions = []
    for c in marks:
        directions += [SearchDirection(c, qr_raw.shape)]
    missing_direction = SearchDirection(estimated_point, qr_raw.shape)
    missing_vline = None
    missing_hline = None
    pts = []
    dest_pts = []
    MARGIN = 20

    for dir in directions:
        extraction = missing_direction.inline_with(dir)
        signs = dir.as_int()
        point = dir.center_point()
        cv2.rectangle(intersected,
                    (point[0] - finder_width//2, point[1] - finder_width//2),
                    (point[0] + finder_width//2, point[1] + finder_width//2),
                    (255, 255, 0), 1)

        roi = binary[
            point[1] - finder_width // 2 : point[1] + finder_width // 2,
            point[0] - finder_width // 2 : point[0] + finder_width // 2,
        ]

        # find edges
        v_edges = []
        h_edges = []
        if signs[1] < 0:
            # low to high
            y_start = 0
            y_stop = roi.shape[0]
        else:
            y_start = roi.shape[0] - 1
            y_stop = 0
        if signs[0] < 0:
            x_start = 0
            x_stop = roi.shape[1]
        else:
            x_start = roi.shape[1] - 1
            x_stop = 0

        # iterate across each row and search for the first black pixel encountered
        for y in range(y_start, y_stop, -signs[1]):
            row = roi[y]

            for x in range(x_start, x_stop, -signs[0]):
                p = row[x]
                if p < BW_THRESH:
                    # NOTE SWAP OF X & Y
                    v_edges += [(
                        y + (dir.center_point()[1] - finder_width // 2),
                        x + (dir.center_point()[0] - finder_width // 2),
                    )]
                    break
        for p in v_edges:
            cv2.line(intersected, (p[1], p[0]), (p[1], p[0]), (0, 0, 255), 1)
        v_fit = best_fit_line_with_outlier_rejection(v_edges, 1.0, 10)
        print(len(v_edges))
        print(v_fit)
        for y in range(0, intersected.shape[0]):
            x = int(v_fit[0] * y + v_fit[1])
            if x >= 0 and x < intersected.shape[1]:
                cv2.line(intersected, (x, y), (x, y), (255, 0, 255), 1)
        if extraction == "v":
            missing_vline = v_fit

        # iterate across each column and search for the first black pixel encountered
        for x in range(x_start, x_stop, -signs[0]):
            col = roi[:,x]
            for y in range(y_start, y_stop, -signs[1]):
                p = col[y]
                if p < BW_THRESH:
                    h_edges += [(
                        x + (dir.center_point()[0] - finder_width // 2),
                        y + (dir.center_point()[1] - finder_width // 2)
                    )]
                    break
        for p in h_edges:
            cv2.line(intersected, p, p, (0, 255, 0), 1)
        h_fit = best_fit_line_with_outlier_rejection(h_edges, 1.0, 10)
        print(len(h_edges))
        print(h_fit)
        for x in range(0, intersected.shape[1]):
            y = int(h_fit[0] * x + h_fit[1])
            if y >= 0 and y < intersected.shape[0]:
                cv2.line(intersected, (x, y), (x, y), (255, 0, 255), 1)
        if extraction == "h":
            missing_hline = h_fit

        pt = pt_from_hv_line(h_fit, v_fit)
        pts += [pt]
        dest_pts += [dir.dest_point(MARGIN)]
        draw_cross(intersected, pt, (255, 255, 0))


    pts += [pt_from_hv_line(missing_hline, missing_vline)]
    print(f"estimated {pts[3]}")
    draw_cross(intersected, pts[3], (255, 255, 0))
    dest_pts += [missing_direction.dest_point(MARGIN)]

    # plot edge points for debugging
    blended_intersection = cv2.addWeighted(intersected, 1.0, rgb, 0.5, 0)
    cv2.imshow('intersection', cv2.flip(blended_intersection, 0))
    cv2.waitKey(0)

    src_pts = np.array(pts, dtype=np.float32)
    dst_pts = np.array(dest_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    ximage = cv2.warpPerspective(qr_raw, H, qr_raw.shape[:2])
    # h = compute_homography(pts, dest_pts)
    # ximage = transform_image(qr_raw, h, qr_raw.shape)
    cv2.imshow('transformed', cv2.flip(ximage, 0))
    cv2.waitKey(0)

    if False:
        # find top left, top right marks. This means we are searching QR codes
        # "right side up". A rotation transform can be inserted that "right side ups"
        # them if they are detected to be rotated.
        tl = None
        tr = None
        for c in marks:
            if c[0] < 64 and c[1] < 64:
                tl = c
            if c[0] > 64 and c[1] < 64:
                tr = c

        if tl is None or tr is None:
            print("couldn't find top left or top right")
            exit(0)

        angle = atan2(tr[1] - tl[1], tr[0] - tl[0])
        tform = AffineTransform(
            a = cos(angle),
            b = -sin(angle),
            c = sin(angle),
            d = cos(angle),
            tx = 0,
            ty = 0,
        )

        affined = affine_transform(binary, tform)
        cv2.imshow('affine', affined)
        cv2.waitKey(0)

        # version number extraction