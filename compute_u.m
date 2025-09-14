function u = compute_u( ...
    static, seg, unit_disk, vert, map, landmark, ...
    op, gaussian_params, k, gamme, delta, target_color, background_color ...
)
    [m, n] = size(static);
    % if vert == init_map
    %     % method = 1;
    %     moving = init_moving;
    % else
    %     % method = 2;
    %     moving = Tools.move_pixels(init_moving, vert, map);
    %     moving = c1 * (moving >= mid) + c2 * (moving < mid);
    % end
    % moving = target_color * init_moving + background_color * (1 - init_moving);
    % moving = Tools.move_seg(unit_disk,vert,map,target_color,background_color);
    moving = seg;
    [ux, uy] = solve_u(static, moving, map, vert, op, landmark, k, gamme, delta);

    if gaussian_params(1) > 0
        ux = reshape(ux, [m, n]);
        uy = reshape(uy, [m, n]);
        ux = gaussian_params(2) * imgaussfilt(ux, gaussian_params(1));
        uy = gaussian_params(2) * imgaussfilt(uy, gaussian_params(1));
        ux = ux(:);
        uy = uy(:);
    else
        ux = gaussian_params(2) * ux;
        uy = gaussian_params(2) * uy;
    end

    u = vert + [ux,uy];
    % map = map + [ux, uy];
%     Fx = scatteredInterpolant(vert, ux);
%     Fy = scatteredInterpolant(vert, uy);
%     map = map+ [Fx(map), Fy(map)];
    % moving = Tools.move_seg(seg,vert,vert+[ux,uy],target_color,background_color);
end
