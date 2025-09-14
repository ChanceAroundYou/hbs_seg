function smooth_mu = smoothing(mu,hbs,op,inner_idx,alpha,beta,lambda,eta,tao,m,n)
% Solve the linear system
% laplacian = Operator.laplacian;
% face_inner_idx = Operator.v2f * inner_idx(:);
% [m,n] = size(mu);
% mu = mu(:);
nu = mu;
iteration = 200;
e0 = nu_energy(nu,mu,hbs,op,alpha,beta,lambda,eta,tao,m,n);
best_e = e0;
best_nu = nu;
not_best_count = 0;

for i = 1:iteration
    if i < 10
        step = 1;
    elseif i < 50
        step = 1;
    else
        step = 0.2;
    end
    
    g = nu_gradient(nu,mu,hbs,op,alpha,beta,lambda,eta,tao,m,n);
    abs_g = max(abs(g));
    if abs_g > 1
        g = g / (20*abs_g);
    elseif abs_g < 1e-5
        break
    end
    nu = nu - g * step;
    e = nu_energy(nu,mu,hbs,op,alpha,beta,lambda,eta,tao,m,n);

    if e < best_e
        best_nu = nu;
        best_e = e;
        not_best_count = 0;
    else
        not_best_count = not_best_count + 1;
        if not_best_count > 15
            break
        end
    end
end
smooth_mu = best_nu;
% nmu = Smooth_Operator\abs(right_hand);
% smooth_mu = nmu.* right_hand ./ abs(right_hand);
end

function e = nu_energy(nu,mu,hbs,op,alpha,beta,lambda,eta,tao,m,n)
    nu_v = op.f2v * nu;
    % nu_m = reshape(nu_v, [m,n]);
    % [gx_nu_m,gy_nu_m] = gradient(nu_m);
    gx_nu_v = op.Diff.Dx * nu_v;
    gy_nu_v = op.Diff.Dy * nu_v;
%     harmonic_term = op.laplacian * angle(nu_v);
    harmonic_term = op.laplacian * log(nu_v);

    e = [norm(nu);
        norm([gx_nu_v,gy_nu_v], 'fro');
        norm(nu - hbs);
        norm(harmonic_term);
        norm(nu - mu)].^2;
    e = [alpha,beta,lambda,eta,tao] * e;
end

function g = nu_gradient(nu,mu,hbs,op,alpha,beta,lambda,eta,tao,m,n)
nu_v = op.f2v * nu;
% nu_m = reshape(nu_v, m, n);
% nu_laplacian = reshape(del2(nu_m), m*n, 1);
% harmonic_term = (1i * reshape(del2(del2(angle(nu_m))), m*n,1)) ./ conj(nu_v);
% harmonic_term(nu_v==0) = 0;
nu_laplacian = op.laplacian * nu_v;
% harmonic_term = (1i * op.double_laplacian * angle(nu_v)) ./ conj(nu_v);
harmonic_term = (op.double_laplacian * log(nu_v)) ./ conj(nu_v);


% harmonic_term2 = reshape(del2(del2(angle(nu_matrix))), m*n,1) .* imag(1 ./ nu);
% harmonic_term2(isnan(harmonic_term2)) = 0;


% g = [nu, reshape(del2(nu_matrix),m*n,1), hbs, mu, harmonic_term];
g = [nu, op.v2f * nu_laplacian, hbs, mu, op.v2f * harmonic_term];
g = g * [alpha + lambda + tao; -beta; -lambda; -tao; eta]; 
end