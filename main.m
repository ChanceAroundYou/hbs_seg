addpath('./dependencies');
addpath('./dependencies/im2mesh');
addpath('./dependencies/mfile');
addpath('./dependencies/aco-v1.1/aco');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
boundary_point_num = 200;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('bc_dict', 'var')
    bc_dict = struct();
end

fname = 'img/hbs_seg/tp1.png';
im = Mesh.imread(fname);
bound = Mesh.get_bound(im, boundary_point_num);

[hbs, he, xq, yq, face, vert, face_center] = HBS(bound, 1000);
bc_dict.('a') = struct('bc', hbs, 'x', xq, 'y', yq, 'bound', bound, 'im', im, 'name', fname);

Plot.plot_mu(hbs, face, vert);


