function PINNBCPLOT(x,y,u,v)
%% 繪製BC與取樣點
figure;
plot(x, y, 'k*');
xlabel('x');
ylabel('y');
title('BC sampling plot');
grid on;
%% 繪製BC速度條件
U = sqrt(u.^2+v.^2); %計算合速度
figure
scatter3(x, y, U, 20, U, 'filled');
view(2); colorbar
xlabel('x'); ylabel('y'); zlabel('U');
title('BC condition plot');
grid on;
end