%--------------------------------------------------------------------- init
%
clear;

% N is number of observations in dlog.dat

% logfilename = 'dlog_firstmark.dat'; N = 758;
% logfilename = 'dlog_secondmark.dat'; N = 1159;
logfilename = 'dlog_thirdmark.dat'; N = 1434;
% logfilename = 'dlog.dat'; N = 3351;

%------------------------------------------------------------ data creation
%expected user input noise
u_err = .15;
M = u_err*eye(2);

% true robot position at t = 1
xt(:,1) = [0 0 0]'; dim = 3;  % x = [x y angle]'

% user input at t = 1
u(:,1) = [0 0]';              % u = [speed delta_angle]'

% Landmark locations
L2006 = [20  20 -20 -20;...      
     20 -20  20 -20];

% You also need the following information about the landmark positions: 
% cyan:magenta -1500 -1000 magenta:cyan -1500 1000 magenta:green 0 -1000 green:magenta 0 1000 yellow:magenta 1500 -1000 magenta:yellow 1500 1000 
% 0 -> green 1 -> magenta 2 -> yellow 3 -> blue	 
L = [-15 -15 0 0 15 15;-10 10 -10 10 -10 10];
LID = [3 1 1 0 2 1;1 3 0 1 1 2];
U = M;         % user input noise (set to be equal to expected input noise)

angle = 0;

logfile = true;

if ~logfile
    
    for t=2:N

        % fabricate user input
        u(2,t) = randn;
        if abs(u(2,t)) > 0.4 %  P(steering) = 0.4
            u(2,t) = 0;
        end
        u(1,t) = .5*(1 - u(2,t)/0.4); % high delta_angle --> low speed

        % create noisy user input
        un = U*randn(2,1) +u(:,t);

        % calculate true robot position t+1
        xt(:,t) =  [xt(1,t-1)+ un(1)*cos(xt(3,t-1)) ; ...
            xt(2,t-1)+ un(1)*sin(xt(3,t-1)) ; ...
            xt(3,t-1)+ un(2)];

    end

    %------------------------------------------------------------- measurements
    %
    perc = .7; % percentage of Landmark measurement loss
    t = 1;
    for i=1:N
        for landmark=1:size(L,2)
            if rand > perc
                % z = [distance angle]'
                z(:,t,landmark) = [ sqrt((L(1,landmark)-xt(1,t))^2 + (L(2,landmark)-xt(2,t))^2)+randn*m_err;...
                    atan2(L(2,landmark)-xt(2,t),L(1,landmark)-xt(1,t)) - xt(3,t)+randn*m_err];
            else
                z(:,t,landmark) = [0;0];
            end
        end
        t = t + 1;
    end

else % logfile

	fid = fopen(logfilename,'r');
    t = 0;
    for i=1:N
        tline = fgetl(fid);
        [type,success] = sscanf(tline, '%s', 1);
        if strcmp(type,'mark')
            fprintf(1,'*')
            continue
        end
        t = t + 1;
        [xt(:,t),success] = sscanf(tline, 'obs: %*d %f %f %f', 3);
        xt(1,t)=xt(1,t)/100; % milimeters to decimeters
        xt(2,t)=xt(2,t)/100; 
        xt(3,t)=xt(3,t)*pi/180; % degrees to radians
        if t > 1
            dx = xt(1,t)-xt(1,t-1);
            dy = xt(2,t)-xt(2,t-1);

            u(2,t) = xt(3, t)- xt(3, t-1); % diff_angle
            u(1,t) = sqrt(dx*dx + dy*dy); % speed
        end
        for landmark=1:6
            z(:,t,landmark) = [0;0];
        end

        [obs_landmarks, success,errmsg,nextindex] = sscanf(tline, 'obs: %*d %*f %*f %*f %d', 1);
        for observation=1:obs_landmarks
            tline=tline(1,nextindex:size(tline,2));
            [signature, success] = sscanf(tline, ' ( %d:%d', 2);
            for landmark = 1:6
                if signature(1) == LID(1,landmark) && signature(2) == LID(2,landmark)
                    [z(:,t,landmark),success,errmsg,nextindex] = sscanf(tline, ' ( %*d:%*d %f %f )', 2);
                    z(1,t,landmark) = z(1,t,landmark) / 100; % milimeters to decimeters
                    z(2,t,landmark) = z(2,t,landmark) * pi / 180; % degrees to radians
                end
            end % for landmarks
        end % for observations
    end % for t=1:N
    fclose(fid);
end % if logfile
				
N = t;
NK = 6; % number of landmarks

% -----------------------------------------------------------------------
% EKF SLAM
%


Sigma = zeros(3 + 2*NK, 3 + 2*NK, N);
Sigma(4:end, 4:end, 1) = eye(2*NK)*10^9;

mu = [xt; zeros(2 * NK,N)];

for i=1:NK
    mu(3 + i*2-1, 1) = mu(1, 1) + z(1, 1, i)*cos(z(2, 1, i) + mu(3, 1));
    mu(3 + i*2, 1) = mu(2, 1) + z(1, 1, i)*sin(z(2, 1, i) + mu(3, 1));
end

for t = 2:N
    %----------------------------------------------------------- prediction
    %3

    %get user input
    v = u(1,t); % velocity
    omega = u(2,t) + 10^-10;	% delta angle
    x = mu(1:3, t-1);
    
    Fx = [eye(3), zeros(3, 2*NK)];
    
    mu_ = mu(:, t-1) + Fx' * [-v/omega * sin(x(3)) + v/omega * sin(x(3)+omega);...
                            v/omega * cos(x(3)) - v/omega * cos(x(3)+omega);...
                            omega];
                
    G = eye(2*NK + 3) + Fx' * [...
        0, 0, -v/omega * cos(x(3)) + v/omega * cos(x(3)+omega);...
        0, 0, -v/omega * sin(x(3)) + v/omega * sin(x(3)+omega);...
        0, 0, 0] * Fx;

    Sigma_ = G * Sigma(:,:,t-1) * G';
    
    
    M = eye(2) * 10^-9;
    V = [cos(mu_(3)+omega), -v*sin(mu_(3)+omega);...
         sin(mu_(3)+omega),  v*cos(mu_(3)+omega);...
                    0,               1];
    R = V*M*V';
    
    Sigma_ = Sigma_ + Fx' * R * Fx;
   
    %----------------------------------------------------------- correction
    %
    for landmark = 1:size(z,3)
        if z(1, t, landmark) ~= 0 % if landmark not measured
            mu_(3 +2*(landmark-1) + 1) = mu_(1) + z(1, t, landmark)*cos(z(2, t, landmark) + mu_(3));
            mu_(3 +2*(landmark-1) + 2) = mu_(2) + z(1, t, landmark)*sin(z(2, t, landmark) + mu_(3));
        
            Q = diag([.15*z(1, t, landmark), .10]+10^-9);

            delta = [mu_(3 +2*(landmark-1) + 1) - mu_(1); mu_(3 +2*(landmark-1) + 2) - mu_(2)];
            q = delta'*delta + 10^-9;

            z_ = [sqrt(q); atan2(delta(2), delta(1)) - mu_(3)];

            Fxj = createF(landmark, NK);

            H = 1/q * [-sqrt(q)*delta(1), -sqrt(q) * delta(2), 0, sqrt(q)*delta(1), sqrt(q) * delta(2);
                delta(2), -delta(1), -q, -delta(2), delta(1)] * Fxj;

            K = Sigma_ * H' / (H * Sigma_ * H' + Q);

            mu_ = mu_ + K * (z(:,t,landmark) - z_);
            Sigma_ = (eye(2*NK+3) - K*H)*Sigma_;
        end
    end
    
    mu(:,t) = mu_;
    Sigma(:,:,t) = Sigma_;
end

close all

for i = 1:10:N
    clf 
    hold on;
    scatter(mu(1, 1:i), mu(2, 1:i), 10, 'filled', 'black');  
    for j=1:NK
        scatter(mu(3+j*2-1, i), mu(3+j*2, i), 25, j, 'filled'); 
        scatter(L(1,j),L(2,j), 25, j, 'filled', 'MarkerEdgeColor', 'black');
        h = plot_gaussian_ellipsoid(mu(3+j*2-1:3+j*2, i), Sigma(3+j*2-1:3+j*2, 3+j*2-1:3+j*2, i));
        set(h,'color','b'); 
    end
    xlim([-25, 25]);
    ylim([-20, 20]);
    drawnow
    pause(0.01)
end

% figure();
% scatter(L(1,:),L(2,:), 5, 'red');
% hold on;
% 
% plot(mu(1, :), mu(2, :), 'r')
% scatter(mu(1, :), mu(2, :), 5, 'k');%, 'filled');
% 
% for i=1:10:size(mu, 2)
%     h = plot_gaussian_ellipsoid(mu(1:2, i), Sigma(1:2, 1:2, i));
%     set(h,'color','b'); 
% end
% 
% % for i=4:2:size(mu, 1)
% %     scatter(mu(i, :), mu(i+1, :), 'x');
% %     hold on;
% % end
% 
% for i=4:2:size(mu, 1)
%     scatter(mu(i, end), mu(i+1, end),'x', 'red');
%     h = plot_gaussian_ellipsoid(mu(i:i+1, end), Sigma(i:i+1, i:i+1, end));
%     set(h,'color','b'); 
% end
    
function F = createF(j, N)
    F = zeros(5, 2*N + 3);
    F(1,1) = 1;
    F(2,2) = 1;
    F(3,3) = 1;
    
    F(4,(2*j)+2) = 1;
    F(5,(2*j)+3) = 1;
end
