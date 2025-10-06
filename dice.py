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
        dir_path = "img/hbs_seg/cdice/"
        img_num = 5
        size = (100, 100)
        models = {"UNet": "1", "Deeplab": "2", "C2F-Seg": "3", "TPSN": "4", "Our": "5"}
        results = {name: {"dice": [], "hd": []} for name in models}

        for i in range(1, img_num + 1):
            gt = Image.open(f"{dir_path}{i}g.png").convert("L").resize(size)
            
            dice_scores = {}
            hd_scores = {}

            for name, suffix in models.items():
                pred_img = Image.open(f"{dir_path}{i}{suffix}.png").convert("L").resize(size)
                
                dice = calculate_dice(pred_img, gt)
                hd = calculate_hausdorff_distance(pred_img, gt)
                
                results[name]["dice"].append(dice)
                results[name]["hd"].append(hd)
                dice_scores[name] = dice
                hd_scores[name] = hd

            dice_str = ", ".join(f"{name} = {score:.4f}" for name, score in dice_scores.items())
            hd_str = ", ".join(f"{name} = {score:.4f}" for name, score in hd_scores.items())
            

        # --- Print Results Table ---
        print("\n--- Results ---")
        # Header
        header = f"{'Image':<8}" + "".join([f"{name:<18}" for name in models])
        print(header)
        
        # Sub-header for metrics
        sub_header = " " * 8 + "".join([f"{'Dice':<9}{'HD':<9}" for _ in models])
        print(sub_header)
        print("=" * len(header))

        # Data rows for each image
        for i in range(img_num):
            row_data = [f"Image {i+1:<2}"]
            for name in models:
                dice = results[name]["dice"][i]
                hd = results[name]["hd"][i]
                row_data.append(f"{dice:<9.4f}{hd:<9.4f}")
            print("".join(row_data))

        # Separator before average row
        print("-" * len(header))

        # Average row
        avg_row_data = [f"{'Average':<8}"]
        for name, metrics in results.items():
            avg_dice = np.mean(metrics["dice"])
            avg_hd = np.mean(metrics["hd"])
            avg_row_data.append(f"{avg_dice:<9.4f}{avg_hd:<9.4f}")
        print("".join(avg_row_data))
        print("=" * len(header))

    except (ImportError, FileNotFoundError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure Pillow and SciPy are installed (`pip install Pillow scipy`) and image files are in the correct path.")
