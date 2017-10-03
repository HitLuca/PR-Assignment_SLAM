%--------------------------------------------------------------------- init
%
clear;

% N is number of observations in dlog.dat

logfilename = 'dlog_firstmark.dat'; N = 758;
% logfilename = 'dlog_secondmark.dat'; N = 1159;
% logfilename = 'dlog_thirdmark.dat'; N = 1434;
% logfilename = 'dlog.dat'; N = 3351;

% expected user input noise
u_err = .15;
M = u_err*eye(2); 

% expected robot location noise
m_err = .1;
Q = m_err*eye(2); 

%------------------------------------------------------------ data creation
%
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
    for t=1:N
        for landmark=1:size(L,2)
            if rand > perc
                % z = [distance angle]'
                z(:,t,landmark) = [ sqrt((L(1,landmark)-xt(1,t))^2 + (L(2,landmark)-xt(2,t))^2)+randn*m_err;...
                    atan2(L(2,landmark)-xt(2,t),L(1,landmark)-xt(1,t)) - xt(3,t)+randn*m_err];
            else
                z(:,t,landmark) = [0;0];
            end
        end
    end

else % logfile

	fid = fopen(logfilename,'r');
    for t=1:N
        tline = fgetl(fid);
        [type,success] = sscanf(tline, '%s', 1);
        if strcmp(type,'mark')
            fprintf(1,'*')
            continue
        end

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
				

NK = 6; % number of landmarks

%---------------------------------------------------------------- a prioris
%
x_ = xt(1:3,1); % a priori x = true robot position
Sigma_ =  0*eye(3*NK + 3); % a priori S = very certain (no error)


%----------------------------------------------------------------------- EKF
%


Sigma = zeros(3 + 2*NK, 3 + 2*NK, N);
Sigma(4:end, 4:end, 1) = eye(2*NK)*10^6;

match = ones(1, NK);

LK = zeros(12,N);

for i =1:N
    LK(:,i) = reshape(z(:,i,:),12,1);
end
mu = [xt; LK];


for t = 2:N
    %----------------------------------------------------------- prediction
    %3

    %get user input
    v = u(1,t);		% velocity
    omega = u(2,t) + 10^-3;	% delta angle
    x = mu(1:3, t);
    
    Fx = [eye(3), zeros(3, 2*NK)];
    
    mu_ = mu(:, t-1) + Fx' * [-v/omega * sin(x(3)) + v/omega * sin(x(3)+omega);...
                            v/omega * cos(x(3)) - v/omega * cos(x(3)+omega);...
                            omega];
                        
                     
                
    G = eye(2*NK + 3) + Fx' * [...
        0, 0, -v/omega * cos(x(3)) + v/omega * cos(x(3)+omega);...
        0, 0, -v/omega * sin(x(3)) + v/omega * sin(x(3)+omega);...
        0, 0, 0] * Fx;

    Sigma_ = G * Sigma(:,:,t-1) * G' + Fx' * Fx; % = G * P_ * G' + Fx' * R * Fx;
    
   
    %----------------------------------------------------------- correction
    %
    for landmark = 1:size(z,3)
        if z(1, t, landmark) == 0 % if landmark not measured???
            mu(3 +2*(landmark-1) + 1) = mu_(1) + z(1, t, landmark)*cos(z(2, t, landmark) + mu_(3));
            mu(3 +2*(landmark-1) + 2) = mu_(2) + z(1, t, landmark)*sin(z(2, t, landmark) + mu_(3));
        end
        
        
        Q = diag([.15*z(1, t, landmark), .10]+10^-3).^2;
        
        mu_(3 +2*(landmark-1) + 1) = mu_(1) + mu(3 +2*(landmark-1) + 1)*cos(mu(3 +2*(landmark-1) + 2) + mu(3));
        mu_(3 +2*(landmark-1) + 2) = mu_(2) + mu(3 +2*(landmark-1) + 1)*sin(mu(3 +2*(landmark-1) + 2) + mu(3));
        
        delta = [mu_(3 +2*(landmark-1) + 1) - mu_(1); mu_(3 +2*(landmark-1) + 2) - mu_(2)];
        q = delta'*delta + 10^-3;
        
        z_ = [sqrt(q); atan2(delta(2), delta(1)) - mu_(3)];
        
        Fxj = createF(landmark, NK);
        
        H = 1/q * [-sqrt(q)*delta(1), -sqrt(q) * delta(2), 0, sqrt(q)*delta(1), sqrt(q) * delta(2);
            delta(2), -delta(1), -q, -delta(2), delta(1)] * Fxj;

       
        K = Sigma_* H' / (H * Sigma_ * H' + Q);
        
        mu_ = mu_ + K * (z(:,t,landmark) - z_);
        Sigma_ = (eye(2*NK+3) - K*H)*Sigma_;
        
        %foundP_(:,:,landmark) = (I-K(:,:,landmark)*H(:,:,landmark))*P_;
    end
    
    mu(:,t) = mu_;
    Sigma(:,:,t) = Sigma_;
end

x = mu;
P = Sigma;

plot(x(1, :), x(2, :), 'r')
hold on;
scatter(x(1, :), x(2, :), 5, 'k', 'filled');

for i=1:5:size(x, 2)
    cov = P(1:2, 1:2, i);
    h = plot_gaussian_ellipsoid(x(1:2, i), P(1:2, 1:2, i), 0.25);
    set(h,'color','b'); 
end

function F = createF(j, N)
    F = zeros(5, 2*N + 3);
    F(1,1) = 1;
    F(2,2) = 1;
    F(3,3) = 1;
    
    F(4,(2*j)+1) = 1;
    F(5,(2*j)+2) = 1;
end
