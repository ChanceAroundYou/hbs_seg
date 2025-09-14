classdef Tools

    methods(Static)
        function norm_value = norm(x, p)
            if nargin == 2 && ~(isreal(p) && all(size(p) == 1))
               error('Type Error, p should be a real scalar')
            elseif nargin == 1
                p = 2;
            end

            if ~isreal(x)
                x = [real(x),imag(x)];
            end
            
            norm_value = sum(x.^p, 2).^ (1/p);
        end
        
        function complex_num = real2complex(real_x, real_y)

            if nargin < 2
                real_y = real_x(:, 2);
                real_x = real_x(:, 1);
            end
            complex_num = real_x + real_y * 1i;
        end
        
        function real_num = complex2real(complex_num)

            real_num = [real(complex_num), imag(complex_num)]; 
        end
        
        function d = circdiff(X)
            d = diff(X);
            d(end+1) = X(1) - X(end);
        end
        
        function d = angle_circdiff(X)
            d = Tools.circdiff(X);
            d(d<-pi) = d(d<-pi)+2*pi;
        end
        
        function w = sqrt_right_half_plane(z)
            w = sqrt(z);
            k = imag(w) .* imag(z) < 0;
            w(k) = -w(k);
        end
        
        function k = near(x, p, delta)
            if nargin < 3
                delta = 1e-6;
            end
            
            k = abs(x-p) < delta;
        end
        
        function k = isinf(x, delta)
            if nargin < 2
                delta = 1e17;
            end
            
            k = abs(x) > delta;
            k(isinf(x)) = 1;
        end
        
        % % % % % % % % % % % % % % % % % % % % % % % % % % % %         
        % mobius
        % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        function w = mobius(z, a, theta)
            if nargin < 3
                k = 1;
            else
                k = exp(theta * 1i);
            end
            
            if a ~= 0
                w = k * (z - a)./(1-conj(a)*z);
                w(isinf(z)) = - k / conj(a);
                w(Tools.near(z, 1/conj(a), 1e-15)) = inf;
            else
                w = k * z;
            end
        end
        
        function w = mobius_inv(z, a, theta)
            if nargin < 3
                w = Tools.mobius(z, -a);
            else
                k = exp(theta * 1i);
                w = Tools.mobius(z, -k*a, -theta);
            end
        end
        
        function d = mobius_d(z, a, theta)
            d = exp(theta*1i) * (1-abs(a)^2)./(1-conj(a)*z).^2;
        end
        
        function f = func_generater(pl)
            syms x y p q;
            zp = (x-p)^2+(y-q)^2;
            ui(p, q) = (p*x^2-p*y^2+2*q*x*y-2*x+p)/zp;
            vi(p, q) = (q*y^2-q*x^2+2*p*x*y-2*y+q)/zp;
            
            u = mean(ui(real(pl), imag(pl)));
            v = mean(vi(real(pl), imag(pl)));
            
            f(x, y) = [u, v];
%             df(x, y) = jacobian([u,v], [x, y]);
        end
        
        function f = func_generater2(pl)
            syms x y p q;
            zp = (x-p)^2+(y-q)^2;
            ui = (p*x^2-p*y^2+2*q*x*y-2*x+p)/zp;
            vi = (q*y^2-q*x^2+2*p*x*y-2*y+q)/zp;
            num = length(pl);
            u = 0;
            v = 0;
            
            for k=1:num
                u = u + subs(ui, [p, q], [real(pl(k)), imag(pl(k))]);
                v = v + subs(vi, [p, q], [real(pl(k)), imag(pl(k))]);
            end
            
            u = u / num;
            v = v / num;
            
            f(x, y) = [u, v];
        end
        
        function f = func_generater3(pl)
            syms x y p q;
            zp = (x-p)^2+(y-q)^2;
            ui = (p*x^2-p*y^2+2*q*x*y-2*x+p)/zp;
            vi = (q*y^2-q*x^2+2*p*x*y-2*y+q)/zp;
            
            u = mean(subs(ui, {p, q}, {real(pl), imag(pl)}));
            v = mean(subs(vi, {p, q}, {real(pl), imag(pl)}));
            
            f(x, y) = [u, v];
%             df(x, y) = jacobian([u,v], [x, y]);
        end
        
        function [x,n] = mulNewton(F, dF,x0)
            x = x0.';
            dx = zeros(2, 1);

            N=100;
            
%             ally=zeros(var_num, N);
%             allx=zeros(var_num, N);

            for n=1:N
                x=x+dx;
                Fx = double(F(x(1), x(2)));
                dFx = double(dF(x(1), x(2)));
                dx=-dFx \ Fx;
                norm_dx = norm(dx);
                if norm_dx > 1
                    dx = dx / (norm_dx * 20);
                
%                 allx(:,n)=x;
%                 ally(:,n)=Fx;
                elseif norm(dx) < eps
                    return
%                     break
                end
            end
        end
        
        function [x, y] = quasiNewton(f, x0)
            options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton', 'StepTolerance', 1e-10,'MaxFunctionEvaluations',1000);
            fun = @(x) double(norm(f(x(1), x(2))));
            [x,y] = fminunc(fun,x0,options);
        end

        function J = move_pixels_old(I, xy, target_xy, grid)
            % move pixels of input image according to transformation of
            % coordinate.
            % INPUT:
            %   I: m x n, input image
            %   xy: mn x 2, original coordinate
            %   target_xy: mn x 2, target coordinate
            if nargin <= 3
                grid = xy;
            end

            [m, n] = size(I);
            % We do this transformation since the y axis is in different
            % order in movepixels
            % xy = reshape(flip(reshape(xy, m, n, 2), 1), m*n, 2);
            % target_xy = reshape(flip(reshape(target_xy, m, n, 2), 1), m*n, 2);
            % target_xy = target_xy - [1,1];
            diff = xy - target_xy;
            diff_x_interpolator = scatteredInterpolant(target_xy,diff(:,1));
            diff_y_interpolator = scatteredInterpolant(target_xy,diff(:,2));
            diff_x = diff_x_interpolator(grid);
            diff_y = diff_y_interpolator(grid);
            Tx = reshape(diff_x,m,n);
            Ty = reshape(diff_y,m,n);

            % We exchange Ty, Tx since grid is created by `ndgrid` in
            % function movepixls
            J = movepixels(I, Ty, Tx);
            % J = flipud(J');
        end
        
        function J = move_pixels2(I, xy, target_xy, grid)
            % move pixels of input image according to transformation of
            % coordinate.
            % INPUT:
            %   I: m x n, input image
            %   xy: mn x 2, original coordinate
            %   target_xy: mn x 2, target coordinate
            if nargin == 3
                grid = xy;
            end

            % We do this transformation since the y axis is in different
            % order in movepixels
            % xy = reshape(flip(reshape(xy, m, n, 2), 1), m*n, 2);
            % target_xy = reshape(flip(reshape(target_xy, m, n, 2), 1), m*n, 2);
            % target_xy = target_xy - [1,1];
            Fx = scatteredInterpolant(target_xy,xy(:,1));
            Fy = scatteredInterpolant(target_xy,xy(:,2));
            ori_grid = [Fx(grid),Fy(grid)];
            G = scatteredInterpolant(grid, I(:));
            J = G(ori_grid);
            J = reshape(J,size(I));
        end
        
        function J = move_pixels(I, xy, target_xy, grid)
            % move pixels of input image according to transformation of
            % coordinate.
            % INPUT:
            %   I: m x n, input image
            %   xy: mn x 2, original coordinate
            %   target_xy: mn x 2, target coordinate
            if nargin == 4
                Fx = scatteredInterpolant(xy,target_xy(:,1));
                Fy = scatteredInterpolant(xy,target_xy(:,2));
                target_grid = [Fx(grid),Fy(grid)];
            elseif nargin == 3
                grid = xy;
                target_grid = target_xy;
            end

            G = scatteredInterpolant(target_grid, I(:));
            J = G(grid);
            J = reshape(J,size(I));
        end

        function J = move_pixels_irregular(I, xy, target_xy, grid)
            % move pixels of input data according to transformation of
            % coordinate. I is not required to be regular rectangle.
            % INPUT:
            %   I: n x 1, input data
            %   xy: n x 2, original coordinate
            %   target_xy: n x 2, target coordinate
            if nargin == 4
                Fx = scatteredInterpolant(xy,target_xy(:,1));
                Fy = scatteredInterpolant(xy,target_xy(:,2));
                target_grid = [Fx(grid),Fy(grid)];
            elseif nargin == 3
                grid = xy;
                target_grid = target_xy;
            end

            G = scatteredInterpolant(target_grid, I);
            J = G(grid);
        end

        function I = irregular2image(data, irxy, xy, m, n)
            F = scatteredInterpolant(irxy, data);
            I = reshape(F(xy), m, n);
        end

        function mu = mu_chop(mu,bound,constant)
            if nargin == 1
                bound = 1-1e-8;
            end
            if nargin < 3
                constant = bound;
            end

            abs_mu = abs(mu);
            idx = find(abs_mu>=bound);
            mu(idx) = constant*(mu(idx)./abs_mu(idx));
        end

        function [seg, high_color, low_color] = move_seg(moving, xy, target_xy, arg1, arg2)
            new_moving = Tools.move_pixels(moving, xy, target_xy);
            
            mid = mean(new_moving, "all");
            high_part = new_moving >= mid;
            low_part = new_moving < mid;

            if nargin == 4
                static = arg1;
                high_color = mean(static(high_part));
                low_color = mean(static(low_part));
                
            else
                high_color = arg1;
                low_color = arg2;
            end
            
            seg = high_color * high_part + low_color * low_part;
        end
        
        function [seg, high_color, low_color] = move_seg_inv(moving, xy, target_xy, arg1, arg2)
            % new_moving = Tools.move_pixels(moving, target_xy, xy);
            [m,n] = size(moving);
            moving_intp = scatteredInterpolant(xy, moving(:));
            new_moving = reshape(moving_intp(target_xy), m,n);
            mid = mean(new_moving, "all");
            high_part = new_moving >= mid;
            low_part = new_moving < mid;

            if nargin == 4
                static = arg1;
                high_color = mean(static(high_part));
                low_color = mean(static(low_part));
                
            else
                high_color = arg1;
                low_color = arg2;
            end
            
            seg = high_color * high_part + low_color * low_part;
        end
        
    end
end

