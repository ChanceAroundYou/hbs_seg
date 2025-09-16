addpath('./dependencies');
addpath('./dependencies/im2mesh');
addpath('./dependencies/mfile');
addpath('./dependencies/aco-v1.1/aco');
addpath('./dependencies/map/map');
clear all;  
close all;

ins_name = 'ins1';
P = struct(...
    'config_name', '1', ...
    'static', "img/hbs_seg/img3.png", ... %'static', 'img/hbs_seg/img3.png', ... % 
    'moving', "img/hbs_seg/tp3.png", ...
    'compare', "img/hbs_seg/img3.c.png",...
    'bound_point_num', 200,...
    'circle_point_num', 1000,...
    'unit_disk_center', [127, 127],...
    'unit_disk_radius', 50,...
    'smooth_eps', 0,...
    'smooth_window', [7, 7],...
    'force_before_nu', 1,...
    'show_mu', 0,...
    'show_results', 0,...
    'upper_bound' , 1.1,...
    'contour_width', 2,...
    'init_image_display',           strcat("img/hbs_seg/output/", ins_name, "/init.png"),...
    'recounstruced_bound_display',  strcat("img/hbs_seg/output/", ins_name, "/reconstructed.png"),...
    'seg_display',                  strcat("img/hbs_seg/output/", ins_name, "/seg_display.png"),...
    'reverse_image', 0,...
    'distort_bound', 1,...
    'iteration',    20,...
    'u_times',      5,...
    'gaussian',     [0, 1],... 
    'alpha',        0.001, ... % abs of mu
    'beta',         0.001, ... % grad of mu
    'gamma',        0.001, ...% abs of f
    'delta',        0.3, ... % grad of f
    'lambda',       0.7, ... % similarity with HBS
    'eta',         0.001, ... % harmonic smooth eta
    'tao',          1, ...% similarity with mu_f
    't_params',     [1.409692e+00 1.583482e+00 4.436754e-02 2.199600e-02] ... % scale, rotation, x, y
);
% [1.491131e+00 6.309427e+00 2.828952e-03 1.312800e-02]; % img1 and tp1
% [8.181691e-01 3.153075e+00 -2.288972e-01 -5.887582e-02]; % img4 and tp2
% [8.100040e-01 1.584319e+00 -5.690997e-02 2.338253e-01]; for img5 and tp2
% [1.246928e+00 4.718931e+00 6.796469e-03 1.757908e-02]; img3 and tp3
% [1.409692e+00 1.583482e+00 4.436754e-02 2.199600e-02]; img2 and tp3
% [1.169707e+00 2.261289e+00 4.818171e-02 3.297825e-02]; img2 and tp1
% [1.328754e+00 3.129071e+00 3.262137e-02 1.372935e-02]; img6 and tp1
% [1.5431402921676636, 1.57, 0.04089602828025818,
% 0.030161170288920403]; img2 and tp5

if endsWith(P.moving, 'mat')
    load(P.moving, 'mean_hbs');
    moving = mean_hbs;
else
moving = Mesh.imread(P.moving);
moving = imresize(moving, [256,256]);
moving = double(moving >= 0.5);
end


static = Mesh.imread(P.static);
static = imresize(static, [256,256]);
% static = double(static >= 0.5);

if P.reverse_image == 1
        static = 1 - static;
end
ref  = static;
static = imnoise(static,"gaussian",0,0.0);
% [PKSNR, SNR] = psnr(static, ref)

% [m,n] = size(static);
% [face,vert] = Mesh.rect_mesh(m,n,0);
% op = Mesh.mesh_operator(face,vert);
% outer_boundary_idx = any([vert(:, 1)==0, vert(:, 1) == (n-1), vert(:,2) == 0, vert(:,2)== (m-1)], 2);
% landmark = find(outer_boundary_idx);

HBS_seg(static,moving, P);
