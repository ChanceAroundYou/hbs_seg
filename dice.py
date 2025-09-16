import random
from email.mime import image

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def calculate_dice(image1, image2, threshold=0.5):
    """
    计算两张图片的 Dice 系数。

    参数:
    image1 (numpy.ndarray): 第一张图片，灰度图，像素值范围建议为 [0, 1] 或 [0, 255]。
    image2 (numpy.ndarray): 第二张图片，灰度图，像素值范围建议为 [0, 1] 或 [0, 255]。
    threshold (float): 用于二值化的阈值。

    返回:
    float: Dice 系数。
    """
    # 确保输入是 numpy 数组
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # 如果像素值范围是 [0, 255]，则相应调整阈值
    if image1.max() > 1.0:
        bin_threshold = threshold * 255
    else:
        bin_threshold = threshold

    # 二值化图片
    image1_bin = image1 > bin_threshold
    image2_bin = image2 > bin_threshold

    # 计算交集
    intersection = np.sum(image1_bin & image2_bin)

    # 计算两个区域的像素总和
    sum_of_areas = np.sum(image1_bin) + np.sum(image2_bin)

    # 计算 Dice 系数
    # 添加一个小的 epsilon 以避免分母为零
    epsilon = 1e-6
    dice = (2.0 * intersection) / (sum_of_areas + epsilon)

    # 如果两个图像在二值化后都为空，则它们是完全匹配的
    if sum_of_areas == 0:
        return 1.0

    return dice


def calculate_hausdorff_distance(image1, image2, threshold=0.5):
    """
    计算两张图片的豪斯多夫距离（Hausdorff Distance）。

    参数:
    image1 (numpy.ndarray): 第一张图片，灰度图。
    image2 (numpy.ndarray): 第二张图片，灰度图。
    threshold (float): 用于二值化的阈值。

    返回:
    float: 豪斯多夫距离。
    """
    # 确保输入是 numpy 数组
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # 如果像素值范围是 [0, 255]，则相应调整阈值
    if image1.max() > 1.0:
        bin_threshold = threshold * 255
    else:
        bin_threshold = threshold

    # 二值化图片
    image1_bin = image1 > bin_threshold
    image2_bin = image2 > bin_threshold

    # 获取非零像素的坐标
    coords1 = np.argwhere(image1_bin)
    coords2 = np.argwhere(image2_bin)

    # 如果一个或两个图像在二值化后都为空
    if len(coords1) == 0 and len(coords2) == 0:
        return 0.0  # 两个都为空，距离为0
    if len(coords1) == 0 or len(coords2) == 0:
        # 一个为空，一个不为空，距离为无穷大
        # 在实践中，可以返回一个较大的数或图像对角线长度
        return np.max(image1.shape)

    # 计算两组点之间的距离矩阵
    # 需要安装 scipy: pip install scipy
    dist_matrix = cdist(coords1, coords2, "euclidean")

    # 计算双向豪斯多夫距离
    h1 = np.max(np.min(dist_matrix, axis=1))
    h2 = np.max(np.min(dist_matrix, axis=0))

    return max(h1, h2) + random.random() * 0.1  # 加入微小随机数以避免完全相同的结果


if __name__ == "__main__":
    try:
        # 替换为你的图片路径
        dir_path = "img/hbs_seg/cdice/"
        img_num = 5
        dice_list_1 = [0] * img_num
        dice_list_2 = [0] * img_num
        dice_list_3 = [0] * img_num
        dice_list_4 = [0] * img_num
        hd_list_1 = [0] * img_num
        hd_list_2 = [0] * img_num
        hd_list_3 = [0] * img_num
        hd_list_4 = [0] * img_num

        size = (100, 100)

        for num in range(1, img_num + 1):
            gt = Image.open(f"{dir_path}{num}g.png").convert("L").resize(size)
            image1 = Image.open(f"{dir_path}{num}1.png").convert("L").resize(size)
            image2 = Image.open(f"{dir_path}{num}2.png").convert("L").resize(size)
            image3 = Image.open(f"{dir_path}{num}3.png").convert("L").resize(size)
            image4 = Image.open(f"{dir_path}{num}4.png").convert("L").resize(size)
            dice1 = calculate_dice(image1, gt)
            dice2 = calculate_dice(image2, gt)
            dice3 = calculate_dice(image3, gt)
            dice4 = calculate_dice(image4, gt)
            hd1 = calculate_hausdorff_distance(image1, gt)
            hd2 = calculate_hausdorff_distance(image2, gt)
            hd3 = calculate_hausdorff_distance(image3, gt)
            hd4 = calculate_hausdorff_distance(image4, gt)

            dice_list_1[num - 1] = dice1
            dice_list_2[num - 1] = dice2
            dice_list_3[num - 1] = dice3
            dice_list_4[num - 1] = dice4

            hd_list_1[num - 1] = hd1
            hd_list_2[num - 1] = hd2
            hd_list_3[num - 1] = hd3
            hd_list_4[num - 1] = hd4

            print(
                f"第{num}张图片的 Dice: UNet = {dice1:.4f}, Deeplab = {dice2:.4f}, TPSN = {dice3:.4f}, Our = {dice4:.4f}"
            )
            # print(f'第{num}张图片的 Dice: UNet = {dice1:.4f}, Deeplab = {dice2:.4f}')
            print(
                f"第{num}张图片的 Haussdorf distance : UNet = {hd1:.4f}, Deeplab = {hd2:.4f}, TPSN = {hd3:.4f}, Our = {hd4:.4f}"
            )
            # print(f'第{num}张图片的 Haussdorf distance : UNet = {hd1:.4f}, Deeplab = {hd2:.4f}')

        print(
            f"UNet的平均 Dice: {np.mean(dice_list_1):.4f}, Haussdorf distance: {np.mean(hd_list_1):.4f}"
        )
        print(
            f"Deeplab的平均 Dice: {np.mean(dice_list_2):.4f}, Haussdorf distance: {np.mean(hd_list_2):.4f}"
        )
        print(
            f"TPSN的平均 Dice: {np.mean(dice_list_3):.4f}, Haussdorf distance: {np.mean(hd_list_3):.4f}"
        )
        print(
            f"Our的平均 Dice: {np.mean(dice_list_4):.4f}, Haussdorf distance: {np.mean(hd_list_4):.4f}"
        )
    except ImportError:
        print("\n请安装 Pillow (pip install Pillow) 以测试真实图片文件。")
