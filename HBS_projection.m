function [map, mu] = HBS_projection(input_map, face,vert, landmark, m, n, P)
sample_num = 300;
bound = smoothResample(input_map(1:P.circle_point_num, :), sample_num);
[hbs, ~, ~, ~, disk_face, disk_vert, ~] = HBS(Tools.real2complex(flipud(bound)), P.circle_point_num, P.unit_disk_radius);
[reconstructed_bound, inner, outer, ~, ~] = HBS_reconstruct(...
    hbs, disk_face, disk_vert, m, n, ...
    P.unit_disk_radius, P.unit_disk_center(1), P.unit_disk_center(2)...
    );



map = [reconstructed_bound; inner; outer];
map = Tools.complex2real(map);

if any(isnan(map))
    error('nan');
elseif any(isinf(map))
    error('inf');
end

reconstructed_bound = Tools.complex2real(reconstructed_bound);
reconstructed_bound = smoothResample(reconstructed_bound, sample_num);

[~, reconstructed_bound, transform] = procrustes(bound, reconstructed_bound);
map = transform.b * map * transform.T + transform.c(1,:);
[tx, ty, scale, theta] = optimize_alignment(bound, reconstructed_bound);
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
map = scale * map * R' + [tx, ty];

mu = bc_metric(face, vert, map, 2);
mu = Tools.mu_chop(mu, P.upper_bound);

% landmark = find(landmark);
% % landmark = [(1:1000)'];
% % idx = randperm(length(lm), 100);
% % lm = lm(idx);
% map2 = lsqc_solver(face, vert, mu, landmark, map(landmark,:));
% mu2 = bc_metric(face, vert, map, 2);
% mu2 = Tools.mu_chop(mu2);
% 1;
% map = lsqc_solver(face, vert, mu, (1:1000)', input_map(1:1000,:));
end
% map = Tools.real2complex(map);
% vert = transform.c(1,:) + transform.b * vert * transform.T;
% vert = Tools.real2complex(vert);
% mu = bc_metric(face, vert, map, 2);





