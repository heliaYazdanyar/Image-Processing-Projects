import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def cal_cost_v(row, col, m, n, cost, nextV, matrix):
    if row == m - 1:
        cost[row][col] = matrix[row][col]
    else:
        if (col == n - 1):
            if ((cost[row + 1][col]) < (cost[row + 1][col - 1])).any():
                cost[row][col] = cost[row + 1][col] + matrix[row][col];
                nextV[row][col] = col
            else:
                cost[row][col] = cost[row + 1][col - 1] + matrix[row][col]
                nextV[row][col] = col - 1

        elif col == 0:
            if (cost[row + 1][col]<cost[row + 1][col + 1]).any():
                cost[row][col] = cost[row + 1][col] + matrix[row][col]
                nextV[row][col] = col

            else:
                cost[row][col] = cost[row + 1][col + 1] + matrix[row][col]
                nextV[row][col] = col + 1

        else:
            if ((cost[row + 1][col]) < (cost[row + 1][col + 1])).any():
                if ((cost[row + 1][col]) < (cost[row + 1][col - 1])).any():
                    cost[row][col] = cost[row + 1][col] + matrix[row][col]
                    nextV[row][col] = col

                else:
                    cost[row][col] = cost[row + 1][col - 1] + matrix[row][col]
                    nextV[row][col] = col - 1

            else:
                if ((cost[row + 1][col + 1]) < (cost[row + 1][col - 1])).any():
                    cost[row][col] = cost[row + 1][col + 1] + matrix[row][col]
                    nextV[row][col] = col + 1
                else:
                    cost[row][col] = cost[row + 1][col - 1] + matrix[row][col]
                    nextV[row][col] = col - 1
    return;


def cal_cost_h(row, col, m, n, cost, nextV, matrix):
    if col == n - 1:
        cost[row][col] = matrix[row][col]

    else:
        if (row == m - 1):
            if ((cost[row][col + 1]) < (cost[row - 1][col + 1])).any():
                cost[row][col] = cost[row][col + 1] + matrix[row][col];
                nextV[row][col] = row
            else:
                cost[row][col] = cost[row - 1][col + 1] + matrix[row][col]
                nextV[row][col] = row - 1

        elif col == 0:
            if ((cost[row][col + 1]) < (cost[row + 1][col + 1])).any():
                cost[row][col] = cost[row][col + 1] + matrix[row][col]
                nextV[row][col] = row

            else:
                cost[row][col] = cost[row + 1][col + 1] + matrix[row][col]
                nextV[row][col] = row + 1

        else:
            if ((cost[row][col + 1]) < (cost[row + 1][col + 1])).any():
                if ((cost[row][col + 1]) < (cost[row - 1][col + 1])).any():
                    cost[row][col] = cost[row + 1][col] + matrix[row][col]
                    nextV[row][col] = row

                else:
                    cost[row][col] = cost[row - 1][col + 1] + matrix[row][col]
                    nextV[row][col] = row - 1

            else:
                if ((cost[row + 1][col + 1]) < (cost[row - 1][col + 1])).any():
                    cost[row][col] = cost[row + 1][col + 1] + matrix[row][col]
                    nextV[row][col] = row + 1
                else:
                    cost[row][col] = cost[row - 1][col + 1] + matrix[row][col]
                    nextV[row][col] = row - 1
    return;


def min_cut_vertical(ssd, rows, cols, cost, nextV):
    i = rows

    for cnt in range(0, rows):
        i = i - 1
        for j in range(0, cols):
            cal_cost_v(i, j, rows, cols, cost, nextV, ssd)


    return


def min_cut_horizontal(ssd, rows, cols, cost, nextV):
    j = cols
    for cnt in range(0, cols):
        j = j - 1
        for i in range(0, rows):
            cal_cost_h(i, j, rows, cols, cost, nextV, ssd)

    return


def ssd(im1, im2):
    s =(im1[:, :, 0] - im2[:, :, 0] )**2
    s1 =(im1[:, :, 1] - im2[:, :, 1])**2
    s2 =(im1[:, :, 2] - im2[:, :, 2])**2


    return (s+s1+s2)


def merge(im1, im2, below):
    if below: a=0
    else: a=1

    b = np.concatenate((im1[:, :, 0], im2[:, :, 0]), axis=a)
    g = np.concatenate((im1[:, :, 1], im2[:, :, 1]), axis=a)
    r = np.concatenate((im1[:, :, 2], im2[:, :, 2]), axis=a)

    res = cv2.merge((b, g, r))

    return res


def get_overlap_R(left,right,cost,next_v):
    res = np.zeros(left.shape)

    val = min(cost[0])
    v = np.where(cost[0]==val)[0][0]
    for i in range(0,p_size):
        for j in range(0, v):
            res[i][j] = 1
        if i < p_size-1:
            v = int(next_v[i][v])

    result = np.where(res == 1, left, right)
    blur = cv2.GaussianBlur(result, (5, 5), 0)

    return blur


def get_overlap_D(up, down, cost, nextV):
    res = np.zeros(up.shape)

    val = min(cost[:, 0])
    v = np.where(cost[:, 0] == val)[0][0]

    for j in range(0, p_size):
        for i in range(0,v):
            res[i][j]=1
        if j < p_size-1:
            v = int(nextV[v][j])

    result = np.where(res == 1, up, down)
    blur = cv2.GaussianBlur(result, (5, 5), 0)

    return blur


def get_applied_L(up_cost, up_nextV, left_cost, left_nextV, black_L, patch):
    res = np.zeros(patch.shape)

    val = np.amin(up_cost[:, 0])
    v = np.where(up_cost[:, 0] == val)[0][0]

    for j in range(0, p_size):
        for i in range(0, v):
            res[i][j] = 1
        if j < p_size - 1:
            v = int(up_nextV[v][j])

    res2 = np.zeros(patch.shape)

    val = min(left_cost[0])
    v = np.where(left_cost[0] == val)[0][0]
    for i in range(0, p_size):
        for j in range(0, v):
            res2[i][j] = 1
        if i < p_size - 1:
            v = int(left_nextV[i][v])

    result = np.where((res == 1) | (res2 == 1), black_L, patch)
    # blur = cv2.blur(result, (5, 5))

    return result


def attach_right(pre, new_loc):
    right = image[new_loc[0]:new_loc[0]+p_size, new_loc[1]:new_loc[1]+p_size, :]
    ssd_right = right[0:p_size, 0:patch_width, :]
    right_2 = right[0:p_size, patch_width:p_size, :]

    left_1 = pre[0:p_size, 0:(pre.shape[1] - patch_width), :]
    ssd_left = pre[0:p_size, (pre.shape[1] - patch_width):pre.shape[1], :]
    ssd_res = ssd(ssd_left, ssd_right)

    cost = np.zeros((p_size,patch_width))
    nextV = np.zeros((p_size,patch_width))

    min_cut_vertical(ssd_res,p_size,patch_width,cost,nextV)

    middle = get_overlap_R(ssd_left, ssd_right, cost,nextV)

    res1 = merge(left_1,middle,False)
    res = merge(res1,right_2,False)


    return res


def find_right(first):
    im = image[0:image.shape[0]-p_size, 0:image.shape[1]-p_size, :]

    template = first[0:p_size, first.shape[1] - patch_width:first.shape[1], :]

    r = cv2.matchTemplate(im, template, cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(r)

    return max_loc


def find_below(mutual_part):
    im = image[0:image.shape[0]-p_size, 0:image.shape[1]-p_size, :]

    r = cv2.matchTemplate(im, mutual_part, cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(r)

    return max_loc


def find_L(L):
    im = image[0:image.shape[0] - p_size, 0:image.shape[1] - p_size, :]

    # img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # temp_gray = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)

    r = cv2.matchTemplate(im, L, cv2.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(r)

    return max_loc


def make_L(black_L):
    res = black_L

    res[patch_width:black_L.shape[0], patch_width:black_L.shape[1], :] = np.random.random((black_L.shape[0]-patch_width, black_L.shape[1]-patch_width,3))*255

    return res


def first_row(firstPatch):
    pre = firstPatch

    x = int((2500-patch_width)/(p_size-patch_width))-1
    for cnt in range(0, x):
        sec = find_right(pre)
        pre = attach_right(pre,sec)

    return pre


def add_first_patch_row(current_img):
    h = current_img.shape[0]
    w = current_img.shape[1]

    black_box = np.zeros(((p_size-patch_width), w, 3),dtype=np.uint8)

    part1 = current_img[h - patch_width:h, 0:p_size, :]

    new_loc = find_below(part1)

    patch = image[new_loc[0]:new_loc[0] + p_size, new_loc[1]:new_loc[1] + p_size, :]
    part2 = patch[0:patch_width, 0:p_size, :]

    ssd1 = ssd(part1, part2)
    cost = np.zeros((patch_width, p_size))
    nextV = np.zeros((patch_width, p_size))
    min_cut_horizontal(ssd1, patch_width, p_size, cost, nextV)

    overlap = get_overlap_D(part1, part2, cost, nextV)

    current_img[h - patch_width:h, 0:p_size, :] = overlap

    black_box[:, 0:p_size, :] = patch[patch_width:p_size, 0:p_size, :]

    res = merge(current_img, black_box, True)

    return res


def add_row(current_img):
    h = current_img.shape[0]
    w = current_img.shape[1]

    black_box = add_first_patch_row(current_img)

    cnt = 1
    x = p_size-patch_width

    while x <= (w-(cnt*x)):
        # getting L-template
        vrt1 = black_box[black_box.shape[0]-p_size:black_box.shape[0], (cnt * x):(cnt * x)+patch_width, :]
        hrz1 = black_box[black_box.shape[0]-x-patch_width :black_box.shape[0]-x, (cnt * x):((cnt * x)+p_size), :]

        # random filling
        L = black_box[black_box.shape[0]-p_size:black_box.shape[0],(cnt * x):(cnt * x)+p_size, :]
        L_prim = make_L(L)

        # getting new patch
        new_loc = find_L(L_prim)
        patch = image[new_loc[0]:new_loc[0]+p_size, new_loc[1]:new_loc[1]+p_size,:]

        hrz2 = patch[0:patch_width, 0:p_size, :]
        vrt2 = patch[0:p_size, 0:patch_width, :]

        left_cost = np.zeros((vrt1.shape[0],vrt1.shape[1]))
        left_nextV = np.zeros((vrt1.shape[0],vrt1.shape[1]))
        ssd1 = ssd(vrt1, vrt2)
        min_cut_vertical(ssd1, p_size, patch_width, left_cost, left_nextV)

        up_cost = np.zeros((hrz1.shape[0],hrz1.shape[1]))
        up_nextV = np.zeros((hrz1.shape[0],hrz1.shape[1]))

        ssd2 = ssd(hrz1, hrz2)
        min_cut_horizontal(ssd2, patch_width, p_size, up_cost, up_nextV)

        black_L = np.zeros((p_size, p_size, 3))

        black_L[:, 0:patch_width, :] = vrt1
        black_L[0:patch_width, :, :] = hrz1

        res = get_applied_L(up_cost, up_nextV, left_cost, left_nextV, black_L, patch)

        black_box[black_box.shape[0]-p_size:black_box.shape[0], (cnt*x):(cnt*x)+p_size, :] = res

        cnt = cnt+1

    return black_box


def build_texture():
    x = random.randint(0, height - p_size)
    y = random.randint(0, width - p_size)

    firstPatch = image[x:x + p_size, y:y + p_size]

    result = first_row(firstPatch)
    x = int((2500 - patch_width) / (p_size - patch_width)) - 1
    for cnt in range(0, x):
        result = add_row(result)

    return result


image1 = cv2.imread('texture1.jpg')
image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

height = image.shape[0]
width = image.shape[1]

p_size = 205+40
patch_width = 40


r2 = build_texture()
blur = cv2.blur(r2, (5, 5))

plt.imshow(blur)
plt.show()

final =cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
cv2.imwrite('res1-2.jpg', final)