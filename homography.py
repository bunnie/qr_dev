"""
Can you give me an implementation in pure python, without relying on cv2 or numpy, of homography finding and transformation? It's ok to use floating point  in this case.
"""
def compute_homography(points_src, points_dst):
    """
    Compute the homography matrix using 4 pairs of source and destination points.

    :param points_src: List of 4 source points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :param points_dst: List of 4 destination points [(x1', y1'), (x2', y2'), (x3', y3'), (x4', y4')].
    :return: 3x3 homography matrix.
    """
    # Building the matrix A for Ah = 0
    A = []
    for (x_src, y_src), (x_dst, y_dst) in zip(points_src, points_dst):
        A.append([
            x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src, -x_dst
        ])
        A.append([
            0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src, -y_dst
        ])

    # Solve using Gaussian Elimination
    h = gaussian_elimination(A)

    # Reshape the solution into a 3x3 matrix
    H = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]]
    return H

def gaussian_elimination(A):
    """
    Solves a system of linear equations using Gaussian elimination.

    :param A: Coefficient matrix augmented with the right-hand side values.
    :return: Solution vector.
    """
    # Number of rows
    n = len(A)

    # Forward Elimination
    for i in range(n):
        # Search for maximum in this column
        max_el = abs(A[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > max_el:
                max_el = abs(A[k][i])
                max_row = k

        # Swap maximum row with current row
        A[i], A[max_row] = A[max_row], A[i]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation for an upper triangular matrix
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]

    return x

def apply_homography(H, point):
    """
    Apply the homography to a point.

    :param H: 3x3 homography matrix.
    :param point: (x, y) point in the source image.
    :return: Transformed (x', y') point.
    """
    x, y = point
    denom = H[2][0] * x + H[2][1] * y + H[2][2]
    x_prime = (H[0][0] * x + H[0][1] * y + H[0][2]) / denom
    y_prime = (H[1][0] * x + H[1][1] * y + H[1][2]) / denom
    return x_prime, y_prime

def transform_image(image, H, output_size):
    """
    Apply homography to an entire image.

    :param image: 2D list representing grayscale image.
    :param H: 3x3 homography matrix.
    :param output_size: (height, width) of the output image.
    :return: Transformed image as a 2D list.
    """
    height, width = output_size
    transformed_image = [[0] * width for _ in range(height)]

    # Iterate over each pixel in the destination image
    for i in range(height):
        for j in range(width):
            # Inverse mapping: find where the pixel (j, i) comes from in the original image
            x_src, y_src = apply_homography(inverse_matrix(H), (j, i))

            # Round to nearest integer for pixel coordinates
            x_src = int(round(x_src))
            y_src = int(round(y_src))

            if 0 <= x_src < len(image[0]) and 0 <= y_src < len(image):
                transformed_image[i][j] = image[y_src][x_src]

    return transformed_image

def inverse_matrix(matrix):
    """
    Compute the inverse of a 3x3 matrix.

    :param matrix: 3x3 matrix.
    :return: Inverse of the matrix.
    """
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

    inv_det = 1.0 / det
    inv = [
        [(matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
         (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
         (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det],
        [(matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
         (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
         (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det],
        [(matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
         (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
         (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det]
    ]
    return inv
