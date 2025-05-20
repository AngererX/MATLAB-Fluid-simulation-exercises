clc;clear
% Find u(x,y),v(x,y),p(x,y)，3 unknowns correspond to 3 equations
% Governing Equations:
% dudx+dvdy=0,
% rho(u dudx + v dudy) = -dpdx + mu (d2udx2 + d2udy2)
% rho(u dvdx + v dvdy) = -dpdy + mu (d2vdx2 + d2vdy2)
%% parameter setting
rho = 1;   % Fluid Density
mu = 0.01; % Dynamic viscositya
U_0 = 1;   % Fluid velocity on lid
L = 1;     % Boundary length
Re = rho*U_0*L/mu; % Re = 100
disp("Re=" + Re);
bcNdata = 250; % Boundary sampling resolution
xb_top = linspace(0,L,bcNdata)';  yb_top = L*ones(bcNdata,1);
xb_bot = linspace(0,L,bcNdata)';  yb_bot = zeros(bcNdata,1);
yb_left = linspace(0,L,bcNdata)'; xb_left = zeros(bcNdata,1);
yb_right= linspace(0,L,bcNdata)'; xb_right= L*ones(bcNdata,1);
% Combine all boundary points
x_b = [xb_top;  xb_bot;  xb_left;   xb_right];
y_b = [yb_top;  yb_bot;  yb_left;   yb_right];
% The corresponding boundary conditions (Dirichlet)
U0 = [U_0*ones(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1)];
V0 = [zeros(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1)];
% P0 = [zeros(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1); zeros(bcNdata,1)];
%PINNBCPLOT(x_b,y_b,U0,V0) % Plot BC and sampling points (ignore for verification)
%Establish supervised data points for training. In theory, completely random training can help improve model capabilities.
Ncp = 10000;  % PDE Control Area Internal Collocation Points
x_c = L*rand(Ncp,1);   % x∈[0,1]
y_c = L*rand(Ncp,1);   % y∈[0,1]
%figure;plot(x_c,y_c,'k*');grid on ; % View the distribution of training collocation points

%% Pack training data into dlarray
% The data is 1xN ('CB'), C is the feature and B is the data sample
dlX = dlarray(x_c','CB');
dlY = dlarray(y_c','CB');
dlX0 = dlarray(x_b','CB'); % BC0
dlY0 = dlarray(y_b','CB'); % BC0
dlU0 = dlarray(U0','CB');  % BC0
dlV0 = dlarray(V0','CB');  % BC0
%dlP0 = dlarray(P0','CB'); % If the initial conditions of P are unknown, do not fill them in.

%% Function call that defines the loss function
function [loss,gradients] = modelLoss(net,dlX,dlY,dlX0,dlY0,dlU0,dlV0,rho, mu)
XY = cat(1,dlX,dlY);      % Merge X,Y to make 2xN dlarray
UVP = forward(net,XY);    % The forward propagation only accepts one input, which is a 2-dimensional vector
U = UVP(1,:);
V = UVP(2,:);
P = UVP(3,:);
% Predict UVP and split
% First-order partial derivative
dudx = dlgradient(sum(U,'all'),dlX,'EnableHigherDerivatives', true);
dudy = dlgradient(sum(U,'all'),dlY,'EnableHigherDerivatives', true);
dvdx = dlgradient(sum(V,'all'),dlX,'EnableHigherDerivatives', true);
dvdy = dlgradient(sum(V,'all'),dlY,'EnableHigherDerivatives', true);
dpdx = dlgradient(sum(P,'all'),dlX,'EnableHigherDerivatives', true);
dpdy = dlgradient(sum(P,'all'),dlY,'EnableHigherDerivatives', true);
% Second-order partial derivatives
d2udx2 = dlgradient(sum(dudx,'all'),dlX,'EnableHigherDerivatives', true);
d2udy2 = dlgradient(sum(dudy,'all'),dlY,'EnableHigherDerivatives', true);
d2vdx2 = dlgradient(sum(dvdx,'all'),dlX,'EnableHigherDerivatives', true);
d2vdy2 = dlgradient(sum(dvdy,'all'),dlY,'EnableHigherDerivatives', true);

% Calculate the mean square error under PDE supervision, pay attention to DL-array
% Verify whether the control equation is consistent? If not, calculate the mean square error
% Find u(x,y),v(x,y),p(x,y), 3 unknowns corresponding to 3 equations
% The original governing equations are：
% dudx+dvdy=0
% rho(u dudx + v dudy) = - dpdx + mu (d2udx2 + d2udy2)
% rho(u dvdx + v dvdy) = - dpdy + mu (d2vdx2 + d2vdy2)
% Let the above equation = 0, error = 0-GE->GE = error
% 0 = rho (u dudx + v dudy) + dpdx - mu (d2udx2 + d2udy2);
% 0 = rho (u dvdx + v dvdy) + dpdy - mu (d2vdx2 + d2vdy2);
GE1 = dudx+dvdy; 
GE2 = rho*(U.*dudx + V.*dudy) + dpdx - mu*(d2udx2 + d2udy2);
GE3 = rho*(U.*dvdx + V.*dvdy) + dpdy - mu*(d2vdx2 + d2vdy2);
mseF = mean(GE1.^2+GE2.^2+GE3.^2);
% Calculate the mean square error with BC
XY0 = cat(1,dlX0,dlY0);
UVP0 = forward(net,XY0); % Substitute the boundary conditions
U0Pred = UVP0(1,:);
V0Pred = UVP0(2,:);
%P0Pred = UVP0(3,:);
% Insert a pressure BC point located in the lower left corner( To ensure numerical stability)
x_pfix = 0; y_pfix = 0;
dlXfix = dlarray(x_pfix,'CB');
dlYfix = dlarray(y_pfix,'CB');
UVPfix = forward(net, [dlXfix; dlYfix]);
Pfix = UVPfix(3,:);
loss_pfix = l2loss(Pfix, dlarray(0,'CB'));

mseU = l2loss(U0Pred,dlU0)+l2loss(V0Pred,dlV0)+loss_pfix;
% Calculate the total mean square error
loss = mseF + mseU;
% Calculate the descent gradient
gradients = dlgradient(loss,net.Learnables);
end

%% Building a Neural Network
numBlocks = 8;     % There are 8 hidden layers
fcOutputSize = 20; % Each layer has 20 neurons
fcBlock = [fullyConnectedLayer(fcOutputSize) 
    tanhLayer]; % Neuron and activation function cascade mode
% featureInputLayer(2) means the input feature dimension (channel) is 2, and the input dlarray should be 2xN
layers = [
    featureInputLayer(2)          % This means that each input data is a 2-dimensional vector, corresponding to 2 neurons
    repmat(fcBlock,[numBlocks 1]) % Number of hidden layers *8
    fullyConnectedLayer(3)] ;     % Number of output layers (3 neurons) = total number of outputs (number of solutions)
net = dlnetwork(layers);          % Encapsulate the neural network and wrap it up as a whole

%% Training solver definition
solverState = lbfgsState; % Storing the state of L-BFGS
maxIterations = 30000;    % Maximum number of training sessions
gradientTolerance = 1e-5; % Gradient Tolerance
stepTolerance = 1e-5;     % Convergence determination Tolerance
accfun = dlaccelerate(@modelLoss);  % Acceleration function calculation (loss function)
%Error convergence monitoring
monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");
iteration = 0;
lossFcn = @(net) dlfeval(accfun,net,dlX,dlY,dlX0,dlY0,dlU0,dlV0,rho, mu);
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);
    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);
    recordMetrics(monitor,iteration,TrainingLoss=solverState.Loss);
    monitor.Progress = 100*iteration/maxIterations;
    if mod(iteration, 50) == 0
        x_c = L * rand(Ncp,1);
        y_c = L * rand(Ncp,1);
        dlX = dlarray(x_c','CB');
        dlY = dlarray(y_c','CB');
        disp('sample changed')
    end
    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end
end

% Defines the drawing grid resolution
Nx = 100;
Ny = 100;
xv = linspace(0,L,Nx);
yv = linspace(0,L,Ny);
[XX,YY] = meshgrid(xv,yv);

% Convert 2D grid coordinates to 2xN input format
gridX = reshape(XX,1,[]);
gridY = reshape(YY,1,[]);
XY = dlarray([gridX; gridY],'CB'); 

% Network forward prediction
UVP_pred = forward(net,XY);
U_pred = reshape(extractdata(UVP_pred(1,:)),Ny,Nx); % u Component
V_pred = reshape(extractdata(UVP_pred(2,:)),Ny,Nx); % v Component
P_pred = reshape(extractdata(UVP_pred(3,:)),Ny,Nx); % Pressure

% calculate the resultant velocity U = sqrt(u^2 + v^2)
U_magnitude = sqrt(U_pred.^2 + V_pred.^2);
figure;
contourf(XX,YY,U_magnitude,30,'LineColor','none');
colorbar;
title('U = \surd(u^2 + v^2) Velocity Contour Map');
xlabel('x'); ylabel('y');
axis equal tight;

figure;
contourf(XX,YY,P_pred,30,'LineColor','none');
colorbar;
title('P Pressure Contour Map');
xlabel('x'); ylabel('y');
axis equal tight;

figure;
quiver(XX,YY,U_pred,V_pred);
title('Velocity Vector Field u,v');

% Define the grid range and resolution for drawing
nx = 50;
ny = 50;
x_min = 0; x_max = L;  % Must be consistent with the simulation range
y_min = 0; y_max = L;

xv = linspace(x_min, x_max, nx);
yv = linspace(y_min, y_max, ny);
[XX, YY] = meshgrid(xv, yv);

% Organize 2D grid data into 2xN dlarray format for neural network input
gridX = reshape(XX,1,[]);
gridY = reshape(YY,1,[]);
XY = dlarray([gridX; gridY], 'CB');

% MAKE Network forward prediction to obtain u(x,y), v(x,y)
UVP = predict(net, XY);
u_pred = reshape(extractdata(UVP(1,:)), ny, nx);
v_pred = reshape(extractdata(UVP(2,:)), ny, nx);

% Streamline Plot
figure;
streamslice(XX, YY, u_pred, v_pred);
title('Predicted velocity field (streamline plot)');
xlabel('x'); ylabel('y');
xlim([x_min x_max]); ylim([y_min y_max]);
axis equal tight;