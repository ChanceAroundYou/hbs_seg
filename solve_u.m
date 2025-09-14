function [ux, uy] = solve_u(I, J, map, vert, Op, boundary_pos, k, gamma, delta)
    [m, n] = size(I);
    num = m * n;

    E = speye(num);
    % res_x = map(:,1) - vert(:,1);
    % res_y = map(:,2) - vert(:,2);
    
%     FI = scatteredInterpolant(vert, I(:));
%     Imap = FI(map);
    
%     I = reshape(Imap, m, n);
%     J = colored_unit_disk;
    
%     [gIx, gIy] = gradient(I);
    % [gJx, gJy] = gradient(J);
%     intp_diff = scatteredInterpolant(vert, I(:)-J(:));
%     diff = intp_diff(map);
    gx = -Op.Diff.Dxv * J;
    gy = -Op.Diff.Dyv * J;
    diff = J(:) - I(:);
    
    

    A = spdiags(k * gx.^2, 0, num, num) + gamma * E - delta * Op.laplacian;
    % B = spdiags(0 * gx .* gy, 0, num, num);
    C = spdiags(k * gy.^2, 0, num, num) + gamma * E - delta * Op.laplacian;

    s = - k * diff .* gx ;
    t = - k * diff .* gy ;
    % M = [A B ; B C];
    % f = [s ; t];
    ux = solveAXB_SP(A, s, boundary_pos);
    uy = solveAXB_SP(C, t, boundary_pos);
    ux(isnan(ux)) = 0;
    uy(isnan(uy)) = 0;

    % u = solveAXB_SP(M, f, [boundary_pos;boundary_pos]);
    % u(isnan(u)) = 0;
    
    % ux = u(1:num);
    % uy = u(num+1:2*num);
end