function hbs_interp_v = compute_hbs_from_image( ...
    fname, unit_disk_radius, unit_disk_center_x, unit_disk_center_y, isplot...
)
% OUTPUT:
%   hbs_interp_v: m x 1 complex, m == height * width
%   vert: m x 2 real, corresponding vert coordinate in form of (x, y)
boundary_point_num = 200;
circle_point_num = 1000;

% size of image
im = Mesh.imread(fname);
[height, width] = size(im);

%%%%%%% IMPORTANT SETTING %%%%%%%
% the size of unit disk
% e.g. it means the radius is 50 pixels now.

% the center of unit disk 
% e.g. it means the disk center into the middel of image.
%%%%%%% IMPORTANT SETTING %%%%%%%
if nargin <= 1
    unit_disk_radius = 50;
end
if nargin <= 2
    unit_disk_center_x = width/2;
    unit_disk_center_y = height/2;
end
if nargin <= 4
    isplot = 1;
end

%% Compute raw HBS, now it is only defined on unit disk.
density = unit_disk_radius;
bound = Mesh.get_bound(im, boundary_point_num);
[hbs, ~, ~, ~, disk_face, disk_vert, ~] = HBS(bound, circle_point_num, density);


%% Extend HBS from unit disk to rectangle.
[extend_face, extend_vert] = Mesh.rect_mesh_from_disk(disk_face, disk_vert, height,width, density, unit_disk_center_x, unit_disk_center_y);
extend_vert = extend_vert * density + [unit_disk_center_x, unit_disk_center_y];
extend_hbs = zeros(size(extend_face,1),1);
extend_hbs(1:size(disk_face,1)) = hbs;


%% Interplate HBS so that it has the same number of points as image.
extend_op = Mesh.mesh_operator(extend_face, extend_vert);
hbs_v = extend_op.f2v * extend_hbs;
[face, vert] = Mesh.rect_mesh(height, width, 0);
interp_map = scatteredInterpolant(extend_vert, hbs_v);
hbs_interp_v = interp_map(vert);
%% (Optial, pull back HBS into each face)
% op = Mesh.mesh_operator(face, vert);
% hbs_interp_f = op.v2f * hbs_interp_v;


if isplot
% disp(fname);
% Plot.imshow(im);
% Plot.plot_mu(hbs, disk_face, disk_vert);
% Plot.plot_mu(extend_hbs, extend_face, extend_vert);
% Plot.plot_map(hbs_interp_v, vert);


% Plot.plot_mu(hbs_interp_f, face, vert);

% hbs_interp_v = Tools.complex2real(reshape(hbs_interp_v, height, width));

rect_hbs = abs(reshape(hbs_interp_v, height, width));
h = round((height - 128) / 2);
w = round((width - 128 ) / 2);
Plot.imshow(rect_hbs(h:height-h, w:width-w));
colormap jet;
end
end