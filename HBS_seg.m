function [map, mu, seg, moving] = HBS_seg(static, moving, P)
    %
    %
    % This program computes the segmentation of an input image based on the QC model.
    % Only input argument "static" is essential.
    %
    % Input:
    % static             m x n, grayscale input image in [0, 1]
    % moving             binary template image
    %
    % Outpt:
    % map                QC registration function from moving to static
    % map_mu             pointwise Beltrami coefficient of the registration
    % seg                deformed moving image

    %% Initializations
    

    bound_point_num = P.bound_point_num;
    circle_point_num = P.circle_point_num;
    center_x = P.unit_disk_center(1);
    center_y = P.unit_disk_center(2);
    center = [center_x, center_y];
    hbs_mesh_density = P.unit_disk_radius;
    smooth_eps = P.smooth_eps;
    mu_upper_bound = P.upper_bound;
    init_image_display = P.init_image_display;
    recounstruced_bound_display = P.recounstruced_bound_display;

    % if isfield(P, 'distort_bound')
    %     distort_bound = P.distort_bound;
    % else
    %     distort_bound = 0;
    % end

    if isfield(P, 'reverse_image') && P.reverse_image
        show_static = 1 - static;
    else
        show_static = static;
    end

    mesh_density = hbs_mesh_density;
    [m, n] = size(static);


    %% Compute HBS and initial map
    if size(moving, 2) ~= 1
        bound = Mesh.get_bound(moving, bound_point_num);
        [hbs, ~, ~, ~, disk_face, disk_vert, ~] = HBS(bound, circle_point_num, hbs_mesh_density);
    else
        hbs = moving;
        circle_interval = (2 / circle_point_num)*pi;
        [disk_face, disk_vert] = Mesh.unit_disk_mesh(hbs_mesh_density, circle_interval);
    end
    
    [reconstructed_bound, inner, outer, normal_vert, face] = HBS_reconstruct(hbs, disk_face, disk_vert, m, n, mesh_density, center_x, center_y);
    normal_face_center = Mesh.get_face_center(face, normal_vert);
    vert = normal_vert .* mesh_density + center;
    hbs_map = Tools.complex2real([reconstructed_bound; inner; outer]);
    hbs_map = hbs_map .* mesh_density + center;

    % interp_map_x = scatteredInterpolant(extend_vert, extend_map(:, 1));
    % interp_map_y = scatteredInterpolant(extend_vert, extend_map(:, 2));
    % normal_map = [interp_map_x(normal_vert), interp_map_y(normal_vert)];
    % hbs_map = normal_map .* mesh_density + center;
    % hbs_mu = bc_metric(face, vert, hbs_map, 2);
    % hbs_mu = Tools.mu_chop(hbs_mu, mu_upper_bound);
    hbs_mu = bc_metric(face, vert, hbs_map, 2);
    hbs_mu = Tools.mu_chop(hbs_mu, mu_upper_bound);
    hbs_mu(Tools.norm(normal_face_center)>1) = 0;
    reconstructed_bound = hbs_map(1:circle_point_num, :);
    % init_moving = double(Tools.move_pixels(unit_disk, vert, map) >= 0.5);

    [rface, rvert] = Mesh.rect_mesh(m, n, 0);
    % init_map = vert;
    
    % normal_vert = (vert - center) ./ mesh_density;
    unit_disk = zeros(size(vert, 1), 1);
    unit_disk(Tools.norm(normal_vert) <= (1 + smooth_eps)) = 1;
    init_moving = unit_disk;

    % Display init_moving, map and mu
    if init_image_display ~= "none"
        figure;
        sp1 = subplot(1, 3, 1);
        imshow(show_static);
        sp2 = subplot(1, 3, 2);
        imshow(Tools.irregular2image(init_moving, vert, rvert, m, n));
        hold on;
        Plot.pri_scatter(reconstructed_bound+[1,1]);
        % plot(real(reconstructed_bound), imag(reconstructed_bound), 'g','LineWidth',1);
        % Plot.pri_plot_mesh(face, map);
        hold off;

        subplot(1, 3, 3);
        Plot.pri_plot_mu(hbs_mu, face, vert);
        colormap(sp1, "gray");
        colormap(sp2, "gray");
        set(gcf, 'unit', 'normalized')
        drawnow();

        if init_image_display ~= "" && endsWith(init_image_display, '.png')

            if ~exist(fileparts(init_image_display), 'dir')
                mkdir(fileparts(init_image_display));
            end

            saveas(gcf, init_image_display);
            fprintf('Saved to %s\n', init_image_display);
        end

    end

    %% Transform initial moving
    if isempty(P.t_params)
        [scaling, rotation, a, b] = get_transformation_params(static, init_moving);
    else
        [scaling, rotation, a, b] = get_transformation_params(static, init_moving, P.t_params);
    end
    updated_map = Tools.complex2real(Tools.real2complex(hbs_map - center)*scaling * exp(1i * rotation))+[a,b]+center;
    updated_moving = Tools.move_pixels(unit_disk,vert,updated_map);
    updated_moving_image = Tools.irregular2image(updated_moving, vert, rvert, m, n);

    if recounstruced_bound_display ~= "none"
        figure;
        subplot(1, 3, 1)
        imshow(show_static);
        subplot(1, 3, 2);
        imshow(updated_moving_image);
        subplot(1, 3, 3);
        imshow(show_static);
        hold on;
        center_pos = updated_map(center_x * n + center_y + 1, :) + [1, 1];
        contour(updated_moving_image, 1, 'EdgeColor', 'g', 'LineWidth', 1);
        Plot.pri_scatter(center_pos);
        hold off;

        set(gcf, 'unit', 'normalized')
        drawnow();

        if recounstruced_bound_display ~= "" && endsWith(recounstruced_bound_display, '.png')

            if ~exist(fileparts(recounstruced_bound_display), 'dir')
                mkdir(fileparts(recounstruced_bound_display));
            end

            saveas(gcf, recounstruced_bound_display);
            fprintf('Saved to %s\n', recounstruced_bound_display);
        end

    end

    %% Compute the object boundary (Main Program)

    % 1st time computation
    [map, mu, seg] = seg_main(static, unit_disk, face, vert, rvert, updated_map, hbs_mu, P);
end

function [scaling, rotation, a, b] = get_transformation_params(static, moving, params)
    s = 256;
    [m, n] = size(static);

    if nargin == 3
        scaling = params(1);
        rotation = params(2);
        a = params(3);
        b = params(4);
    else

        if m ~= n
            static = Mesh.pad2square(static);
            moving = Mesh.pad2square(moving);
        end

        if max(m, n) ~= s
            static = imresize(static, [s, s]);
            moving = imresize(moving, [s, s]);
        end

        static = cast(static * 255, 'uint8');
        moving = cast(moving * 255, 'uint8');

        url = 'http://aws.cheerl.space:8880/get_params';
        data = struct('static', static, 'moving', moving);
        json = jsonencode(data);
        options = weboptions('MediaType', 'application/json');
        options.Timeout = 300;
        response = webwrite(url, json, options);
        scaling = response(1);
        rotation = response(2);
        a = response(3);
        b = response(4);
        fprintf('%d %d %d %d\n', scaling, rotation, a, b)
    end

end
