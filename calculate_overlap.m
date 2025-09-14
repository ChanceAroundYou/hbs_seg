function overlap = calculate_overlap(A, B)
    % 计算两个图形的重叠面积（假设它们的点按顺序排列，且是简单多边形）
    overlap = 0;
    
    % 将 A 和 B 合并成一个单一的多边形
    % 可以使用交集方法（例如，利用`polyshape`类来计算交集）
    poly_A = polyshape(A(:,1), A(:,2));
    poly_B = polyshape(B(:,1), B(:,2));
    
    % 计算交集
    intersection = intersect(poly_A, poly_B);
    
    % 返回交集面积
    overlap = area(intersection);
end