%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Extended Kalman Filter
%   by J�rgen Sturm, Tijn Schmits, Arnoud Visser
%   April 2008
%
% Based on:
%
% Wolfram Burgard's
% http://ais.informatik.uni-freiburg.de/teaching/ss07/robotics/slides/
% --> 09.pdf
%
% Thrun's
% http://robots.stanford.edu/probabilistic-robotics/ppt/slam.ppt
%
% Dataset dlog.dat provided by Steffen Gutmann, 6.5.2004
% http://cres.usc.edu/radishrepository/view-one.php?name=comparison_of_self-localization_methods_continued
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------- init
%
clear;

% N is number of observations in dlog.dat

%logfilename = 'dlog_firstmark.dat'; N = 758;
% logfilename = 'dlog_secondmark.dat'; N = 1159;
 logfilename = 'dlog_thirdmark.dat'; N = 1434;
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
    t = 0;
    for i=1:N
        tline = fgetl(fid);
        [type,success] = sscanf(tline, '%s', 1);
        if strcmp(type,'mark')
            fprintf(1,'*')
            continue
        end
        t = t+1;
        [xt(:,t),success] = sscanf(tline, 'obs: %*d %f %f %f', 3);
        xt(1,t)=xt(1,t)/100; % milimeters to decimeters
        xt(2,t)=xt(2,t)/100; % degrees to radians
        xt(3,t)=xt(3,t)*pi/180;
        if t > 1
            dx=xt(1,t)-xt(1,t-1);
            dy=xt(2,t)-xt(2,t-1);

            u(2,t) = xt(3,t)-xt(3,t-1); % diff_angle
            u(1,t) = sqrt (dx*dx+dy*dy); % speed
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

%---------------------------------------------------------------- a prioris
%
x_ = xt(1:3,1); % a priori x = true robot position
P_ =  0*eye(3); % a priori P = very certain (no error)

%----------------------------------------------------------------------- EKF
 %
x = zeros( dim, N );
P = zeros( dim, dim, N );
I = eye(dim);
match = ones(1, N);

for t = 1:N
    %----------------------------------------------------------- prediction
    %

    %get user input
    v = u(1,t);		% velocity
    da = u(2,t);	% delta angle
   
    % Jacobian with respect to robot location
    G = [1          0 -v*sin(x_(3)+da);...
         0          1  v*cos(x_(3)+da);...
         0          0               1];

    % Jacobian with respect to control
    V = [cos(x_(3)+da) -v*sin(x_(3)+da);...
         sin(x_(3)+da)  v*cos(x_(3)+da);...
                    0               1];
    
    % predicted robot position mean
    x_ = [x_(1) + v*cos(x_(3)+da);...
          x_(2) + v*sin(x_(3)+da);...
                         x_(3)+da];

    % predicted covariance
    P_ = G*P_*G' + V*M*V';

    %----------------------------------------------------------- correction
    %
    for landmark = 1:size(z,3)
        if z(1,t,landmark) ~= 0   % if Landmark is measured

            % predicted measurement
            z_ = [sqrt((L(1,landmark)-x_(1))^2 + (L(2,landmark)-x_(2))^2);...
                atan2(L(2,landmark)-x_(2),L(1,landmark)-x_(1)) - x_(3)];

            % Jacobian of H with respect to location
            H(:,:,landmark) = [ -(L(1,landmark)-x_(1))/(L(1,landmark)^2-2*L(1,landmark)*x_(1)+x_(1)^2+L(2,landmark)^2-2*L(2,landmark)*x_(2)+x_(2)^2)^(1/2), -(L(2,landmark)-x_(2))/(L(1,landmark)^2-2*L(1,landmark)*x_(1)+x_(1)^2+L(2,landmark)^2-2*L(2,landmark)*x_(2)+x_(2)^2)^(1/2),  0;
                (L(2,landmark)-x_(2))/(L(1,landmark)^2-2*L(1,landmark)*x_(1)+x_(1)^2+L(2,landmark)^2-2*L(2,landmark)*x_(2)+x_(2)^2),       -(L(1,landmark)-x_(1))/(L(1,landmark)^2-2*L(1,landmark)*x_(1)+x_(1)^2+L(2,landmark)^2-2*L(2,landmark)*x_(2)+x_(2)^2), -1];
            
            
            Q = diag([.15*z(1, t, landmark), .10]+10^-9);
            % predicted  measurement covariance
            S = H(:,:,landmark)*P_*H(:,:,landmark)' + Q;
            
            %Kalman gain
            K(:,:,landmark) = P_* H(:,:,landmark)' / S;

            %innovation
            nu = z(:,t,landmark) - z_;
            
            %validation gate
            ro = nu'/S*nu; % From Kristensen IROS'03, section III.A
            
            if ro < 2
                %updated mean and covariance
                foundx(:,landmark) = x_ + K(:,:,landmark)*nu;
                foundP_(:,:,landmark) = (I-K(:,:,landmark)*H(:,:,landmark))*P_;
            else
                %propagate known mean and covariance
                foundx(:,landmark) = x_;
                foundP_(:,:,landmark) = P_;
                z(:,t,landmark)=[0; 0];
            end

        else
            %propagate known mean and covariance
            foundx(:,landmark) = x_;
            foundP_(:,:,landmark) = P_;
        end
    end

    % determine mean
    x_ = mean(foundx,2);
    P_ = mean(foundP_,3);

    % create history
    x(:,t) = x_;
    P(:,:,t) = P_;
end

plot(x(1, :), x(2, :), 'r')
hold on;
scatter(x(1, :), x(2, :), 5, 'k', 'filled');

for i=1:5:size(x, 2)
    cov = P(1:2, 1:2, i);
    h = plot_gaussian_ellipsoid(x(1:2, i), P(1:2, 1:2, i), 1/5);
    set(h,'color','b'); 
end
