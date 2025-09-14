function area = polygon_area(A)
    % A 是一个 nx2 的矩阵，表示边界点坐标
    x = A(:, 1); % x 坐标
    y = A(:, 2); % y 坐标
    n = size(A, 1); % 顶点个数
    
    % 使用高斯公式计算面积
    area = 0.5 * abs(sum(x(1:n-1) .* y(2:n) - x(2:n) .* y(1:n-1)) + (x(n) * y(1) - x(1) * y(n)));
end