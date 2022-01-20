final_T   =[2.25      7.68      10.28     2.16      5.47];
final_R   =[0.3084    0.9997    0.8811    0.5329    0.3896];
final_MSE =[0.6526    0.0091    0.3023    0.5800    0.5787];


final_T_noise   =[2.82   7.11     10.63    2.21    15.37];
final_R_noise   =[0.3370 0.9392   0.8384   0.5190  0.3789];
final_MSE_noise =[0.6913 0.2564   0.4185   0.6462  0.6376];

X = categorical({'gd','lm','bfg','gda','br'});
X = reordercats(X,{'gd','lm','bfg','gda','br'});


Y_T   = [final_T;final_T_noise]';
Y_R   = [final_R;final_R_noise]';
Y_mse = [final_MSE;final_MSE_noise]';


figure,bar(X,Y_R,0.3)
ylabel('R')
title('compare R of algorithms')
legend('without noise','with noise')

figure,bar(X,Y_T,0.3)
ylabel('Time')
title('compare time of algorithms')
legend('without noise','with noise')

figure,bar(X,Y_mse,0.3)
ylabel('RMSE')
title('compare RMSE of algorithms')
legend('without noise','with noise')

