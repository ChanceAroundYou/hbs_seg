function B = smoothResample(A, m)
% 对边界点矩阵A进行光滑均匀重采样，生成m个点的矩阵B
% 输入：
%   A - nx2矩阵，原始边界点坐标
%   m - 重采样后的点数
% 输出：
%   B - mx2矩阵，重采样后的边界点

if nargin <= 1
    m = size(A, 1);
end

% 检查闭合性
isClosed = all(abs(A(1,:) - A(end,:)) < 1e-8);
if ~isClosed
    A = [A; A(1,:)]; % 非闭合则闭合处理
end

% 计算累积弧长
diffs = diff(A);
dists = sqrt(sum(diffs.^2, 2));
cumDists = [0; cumsum(dists)];
totalLength = cumDists(end);

% 样条插值设置
t = cumDists;
x = A(:,1);
y = A(:,2);

if isClosed
    % 周期性三次样条插值（闭合曲线）
    ppx = csape(t, x', 'periodic');
    ppy = csape(t, y', 'periodic');
else
    % 保形分段三次插值（非闭合曲线）
    ppx = pchip(t, x');
    ppy = pchip(t, y');
end

% 生成均匀参数
if isClosed
    tNew = linspace(0, totalLength, m + 2)';
    tNew = tNew(1:end-1); % 避免重复首尾点
else
    tNew = linspace(0, totalLength, m+1)';
end

% 计算新点
xNew = ppval(ppx, tNew);
yNew = ppval(ppy, tNew);
B = [xNew(:), yNew(:)];
B = B(1:end-1, :);
end