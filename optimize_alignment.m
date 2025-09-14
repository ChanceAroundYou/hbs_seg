function [tx, ty, scale, theta] = optimize_alignment(A, B)
    % A, B 是两个 n x 2 矩阵，表示两个图形的边界点坐标
    
    % 定义目标函数（负的重叠面积，因为我们要最大化重叠面积）
    obj_func = @(params) -calculate_iou(A, transform_shape(B, params));
    
    % 初始猜测：平移量 [0, 0]，缩放因子为 1，旋转角度为 0
    init_params = [0, 0, 1, 0];
    
    % 使用 fminunc 进行优化，找到平移、缩放和旋转的最佳参数
    % options = optimset('Display', 'iter', 'PlotFcns','optimplotfval','TolFun',2e-6,'TolX',2e-6);
    options = optimset('Display', 'off', 'TolFun',2e-6,'TolX',2e-6);
    optimal_params = fminsearch(obj_func, init_params, options);
    
    % 返回最优的平移、缩放和旋转参数
    tx = optimal_params(1);
    ty = optimal_params(2);
    scale = optimal_params(3);
    theta = optimal_params(4);
end

function B_transformed = transform_shape(B, params)
    % 对矩阵 B 进行平移、缩放和旋转变换
    tx = params(1);        % 平移 x
    ty = params(2);        % 平移 y
    scale = params(3);     % 缩放因子
    theta = params(4);     % 旋转角度（弧度）
    
    % 创建旋转矩阵
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    
    % 对 B 进行变换
    B_transformed = scale * B * R' + [tx, ty];  % 先旋转再平移
end

function iou = calculate_iou(A, B)
    % 计算两个图形的重叠面积（假设它们的点按顺序排列，且是简单多边形）
    % overlap = 0;
    
    % 将 A 和 B 合并成一个单一的多边形
    % 可以使用交集方法（例如，利用`polyshape`类来计算交集）
    poly_A = polyshape(A(:,1), A(:,2));
    poly_B = polyshape(B(:,1), B(:,2));
    
    % 计算交集
    intersection = intersect(poly_A, poly_B);
    intersection_area = area(intersection);

    union_poly = union(poly_A, poly_B);
    union_area = area(union_poly);  % 并集面积
    
    % 计算 IoU
    if union_area > 0
        iou = intersection_area / union_area;
    else
        iou = 0;  % 如果并集面积为 0，则 IoU 也为 0
    end
end

