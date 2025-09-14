classdef Mesh
    %MESH 此处显示有关此类的摘要
    %   此处显示详细说明
    methods(Static)
        function im = imread(path)
            % Read image from path and turn to grayscale image in [0, 1]
            im = imread(path);
            if size(im, 3) == 3
                im = double(rgb2gray(im)) / 255;
            elseif size(im, 3) == 1
                im = double(im) / 255;
            end
        end
        
        function padded_im = pad2square(im)
            shape = size(im);
            [max_size, index] = max(shape);
            min_size = shape(3-index);
            start = floor((max_size-min_size)/2);
            padded_im = zeros(max_size, max_size);
            if index == 1
                padded_im(:, 1:start-1) = mean(im(:, 1));
                padded_im(:, start:start+min_size-1) = im;
                padded_im(:, start+min_size:end) = mean(im(:, end));
            else
                padded_im(1:start-1, :) = mean(im(1, :));
                padded_im(start:start+min_size-1, :) =im;
                padded_im(start+min_size:end, :) = mean(im(end, :));
            end
        end
            
        function bound = get_bound(im, num)
            if nargin == 1
                num = 0;
            end
            if length(size(im)) == 3
                im = rgb2gray(im);
            end
            im = imbinarize(im);

            bounds = im2Bounds(im);
            bounds = getCtrlPnts(bounds, false);
            bounds_size = cellfun(@(x) size(x,1), bounds{1});
            bounds_size(bounds_size == sum(size(im))*2+1) = 0;
            [~, k] = max(bounds_size);
            bound = bounds{1}{k}(1:end-1, :);
            bound(:, 2) = size(im, 1) + 1 - bound(:, 2);
            bound = Tools.real2complex(bound);
            
            sigma = 4;
            gauss_size = 7;
            bound_real = imgaussfilt(real(bound), sigma, 'FilterSize', gauss_size);
            bound_imag = imgaussfilt(imag(bound), sigma, 'FilterSize', gauss_size);
            bound = bound_real + 1i * bound_imag;
            % bound = imgaussfilt(bound, 1); % Adjust sigma for more or less smoothing
     
            if num ~= 0
                bound_size = length(bound);
                interval = (bound_size-1) / (num-1);
                bound_x = (1:bound_size).';
                bound_xq = (1:interval:bound_size).';
                bound = interp1(bound_x, bound, bound_xq, 'linear');
            end
        end

        function [bound, bound_idx] = get_bound2(im, num)
            % INPUT
            %     im: m x n, binary image where 0 means background, 1 means
            %     object
            %     num: int, resample bound to `num` points, optional
            % OUTPUT
            %     bound: object boundary coordinate
            %     bound_idx: the idx of boundary. It's empty when `num` is
            %     given
            if nargin < 2
                num = 0;
            end
            
            im = imbinarize(im);
            checkbd = padarray(im,[1,1]);
            checkbd = checkbd(2:end-1,1:end-2)+checkbd(3:end,2:end-1)+checkbd(2:end-1,3:end)+checkbd(1:end-2,2:end-1);
            checkbd = checkbd.*im;
            bound_idx = intersect(find(checkbd>0),find(checkbd<4));
            [bound_y,bound_x] = ind2sub(size(im),bound_idx);
            bound = [bound_x,bound_y];

            angle = atan2((bound_x-mean(bound_x)),(bound_y-mean(bound_y)));
            [~,order] = sort(angle);
            bound_idx = bound_idx(order);
            bound = bound(order, :);
            

            if num
                bound_size = length(bound);
                interval = (bound_size-1) / (num-1);
                bound = interp1((1:bound_size).', Tools.real2complex(bound), (1:interval:bound_size).', 'linear');
                bound = Tools.complex2real(bound);
                bound_idx = [];
            end
        end

        
        function [face, vert] = rect_mesh(height, width, normal)
            if nargin < 3
                normal = 1;
            end

            [x,y] = meshgrid(0:width-1, 0:height-1);
            % y = y(end:-1:1, :);
            if normal
                x = x/(width-1);
                y = y/(height-1);
            end
            vert = [x(:),y(:)];

            face = delaunay(vert(:,1), vert(:, 2));
        end

        function [face, vert] = unit_disk_mesh(density, interval, eps)
            if nargin == 0
                density = 100;
            end
            if nargin < 2
                interval = 0;
            end
            if nargin < 3
                eps = 0.01;
            end
            
            [x,y] = meshgrid(-density:density, -density:density);
            % y = y(end:-1:1, :);
            vert = [x(:),y(:)] ./ density;
            index = Tools.norm(vert) <= 1 - eps;
            vert = vert(index, :);
            if interval
                circle = Tools.complex2real(exp(1i*(0:interval:2*pi-interval)'));
                vert = [circle;vert];
            end
            face = delaunay(vert(:, 1), vert(:, 2));
        end
        
        function [face, vert, outer_face, outer_vert] = rect_mesh_from_disk(disk_face, disk_vert, height, width, density, center_x, center_y)
            if nargin <= 5
                center_x = width / 2;
                center_y = height / 2;
            end
            circle_point_num = length(disk_vert(abs(Tools.norm(disk_vert) - 1) < 1e-4,:));
            circle_vert = disk_vert(1:circle_point_num, :);

            [x,y] = meshgrid(0:width-1, 0:height-1);
            x = (x - center_x) ./ density;
            y = (y - center_y) ./ density;
            % y = y(end:-1:1, :);
            outer_vert = [x(:),y(:)];
            outer_vert = outer_vert(Tools.norm(outer_vert) > 1, :);

            vert = [disk_vert;outer_vert];
            outer_vert_with_circle = [circle_vert;outer_vert];

            outer_face = delaunay(outer_vert_with_circle(:,1), outer_vert_with_circle(:, 2));
            outer_face_center = Mesh.get_face_center(outer_face,outer_vert_with_circle);
            outer_face = outer_face(Tools.norm(outer_face_center)>1,:);
            outer_face(outer_face>circle_point_num) = outer_face(outer_face>circle_point_num)+length(disk_vert)-circle_point_num;
            face = [disk_face;outer_face];
        end
        
        function face_center = get_face_center(face, vert)
            face_size = size(face, 1);
            face_vert = reshape(vert(face(:, :), :), face_size, 3, 2);
            face_center = reshape(mean(face_vert, 2), face_size, 2);
        end

        function operator = mesh_operator(face,vert)
            operator = meshOperator(vert,face);
        end
    end
end

