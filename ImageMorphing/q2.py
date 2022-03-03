import dlib
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
import imageio


def find_points(image):
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    h = image.shape[0]
    w = image.shape[1]

    # adding points to dlib points
    arr = np.array([[0, 0],
                    [w - 1, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [0, int(h / 2)],
                    [int(w / 2), 0],
                    [w - 1, int(h / 2)],
                    [int(w / 2), h - 1]])

    for (s, rect) in enumerate(rects):
        points = predictor(gray, rect)
        points = face_utils.shape_to_np(points)

        points = np.concatenate((points, arr), axis=0)

    return points


def get_triangles(image,points):
    shape_rect = (0, 0, image.shape[1], image.shape[0])

    sub = cv2.Subdiv2D(shape_rect)
    for p in points:
        sub.insert((p[0], p[1]))

    list = sub.getTriangleList()

    return list


def triangles_indexes(image, points):
    tri_list = get_triangles(image, points)

    triangles = []
    for curr_triangle in tri_list:
        index_of_points = [-1,-1,-1]
        for i in range(0,6,2):
            for index in range(len(points)):
                if (curr_triangle[i] == points[index][0]) & (curr_triangle[i + 1] == points[index][1]):
                    index_of_points[int(i/2)] = index
                    break

        triangles.append(index_of_points)

    return triangles


def get_contour_rect(triangle):
    x, y, w, h = cv2.boundingRect(triangle)

    location = [x, y]
    size = (w, h)

    new_vertexes = []

    for i in range(0,3):
        x0 = triangle[i][0] - x
        y0 = triangle[i][1] - y
        new_vertexes.append((x0, y0))

    return [location, size, new_vertexes]


def warp_affine(img, size, triangle_1, triangle_2):

    affine = cv2.getAffineTransform(np.float32(triangle_1), np.float32(triangle_2))
    warped = cv2.warpAffine(img, affine, size, None, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_DEFAULT)

    return warped


def get_sub_matrix(img, location, size):
    x = location[1]
    w = size[1]

    y = location[0]
    h = size[0]

    sub = img[x:x + w, y:y + h]
    return sub


def get_inside_tri(dst_tri, size, morphed_rect, original_im):
    # getting a binary matrix of dst triangle
    binary_matrix = np.zeros((size[1], size[0], 3), dtype=np.float32)
    cv2.fillConvexPoly(binary_matrix, np.int32(dst_tri), (1, 1, 1), 20, 0)

    triangle_result = morphed_rect * binary_matrix
    outbox = original_im * (1 - binary_matrix)

    return triangle_result + outbox


def morphing(triangle_1, triangle_2, triangle_dst, alpha, first_img, final_img, dst_matrix):
    loc_1, size_1, new_v1 = get_contour_rect(np.float32(triangle_1))
    loc_2, size_2, new_v2 = get_contour_rect(np.float32(triangle_2))
    loc_dst, size_dst, new_v_dst = get_contour_rect(np.float32(triangle_dst))

    sub_1 = get_sub_matrix(first_img,loc_1, size_1)
    warped_1 = warp_affine(sub_1, size_dst, new_v1, new_v_dst)

    sub_2 = get_sub_matrix(final_img, loc_2, size_2)
    warped_2 = warp_affine(sub_2, size_dst, new_v2, new_v_dst)

    sub_result = (1-alpha) * warped_1 + alpha * warped_2

    inside_triangle = get_inside_tri(new_v_dst,size_dst, sub_result,dst_matrix[loc_dst[1]:loc_dst[1]+size_dst[1], loc_dst[0]:loc_dst[0]+size_dst[0]])


    # adding new sub matrix to dst_matrix
    dst_matrix[loc_dst[1]:loc_dst[1]+size_dst[1], loc_dst[0]:loc_dst[0]+size_dst[0]] = inside_triangle

    return


img_start = cv2.imread('a.jpg')
img_final = cv2.imread('b.jpg')


# size
h = img_start.shape[0]
w = img_start.shape[1]


# 1
# facial land marks
points_1 = find_points(img_start)
points_2 = find_points(img_final)

# 2
# delaunay
tri_mutual = triangles_indexes(img_start, points_1)

# setting alphas
alpha_list = []
for i in range(1, 101):
    alpha_list.append(i/100)


# 3
img_list = []
for cnt in range(0, 100):
    alpha = alpha_list[cnt]

    # step a
    points = []
    for i in range(0, len(points_1)):
        x = (1 - alpha) * points_1[i][0] + alpha * points_2[i][0]
        y = (1 - alpha) * points_1[i][1] + alpha * points_2[i][1]
        points.append((x, y))

    # step b
    result = np.zeros(img_start.shape, dtype=np.float32)

    for a, b, c in tri_mutual:
        triangle1 = [points_1[a], points_1[b], points_1[c]]
        triangle2 = [points_2[a], points_2[b], points_2[c]]
        dst_triangle = [points[a], points[b], points[c]]

        morphing(triangle1, triangle2, dst_triangle, alpha, img_start.copy(), img_final.copy(), result)

    # step c
    res = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img_list.append(res)


print("list is ready")

# making gif
imageio.mimsave('res2.gif', img_list)

print("executed")
