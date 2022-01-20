
% final_T =[4.0690    4.0719    9.7229    0.7266    4.7666];
% final_R =[0.6193    0.8885    0.9950    1.0000    0.7477];
% X = categorical({'gd','gda','bfg','lm','br'});
% X = reordercats(X,{'gd','gda','bfg','lm','br'});

% final_T =[69.9445    103.5388];
% final_R =[1.0000 1.0000];
% final_MSE =[2.700637543147569e-04 6.567520462263955e-05];
% X = categorical({'lm','br'});
% X = reordercats(X,{'lm','br'});

% final_T =[3.0745    3.1250    16.56     12.75 ];
% final_R =[0.3886    0.5998    0.9112    0.9972];
% final_MSE =[0.6338 0.5483 0.2608 0.0126];
% X = categorical({'gd','gda','bfg','lm'});
% X = reordercats(X,{'gd','gda','bfg','lm'});

% final_T =[14.2000    5.6936];
% final_R =[1.0000 0.3247];
% final_MSE =[0.0017 0.6158];
% X = categorical({'lm','br'});
% X = reordercats(X,{'lm','br'});

final_T =[136.0743    256.0708];
final_R =[1.0000 1.0000];
final_MSE =[9.2e-05 2.2e-05];
X = categorical({'lm','br'});
X = reordercats(X,{'lm','br'});

% figure,bar(X,(final_MSE),0.2)
% ylabel('MSE')
% title('compare MSE of algorithms')
% 
% figure,bar(X,(final_R),0.2)
% ylabel('R')
% title('compare R of algorithms')
% 
% 
% figure,bar(X,(final_T),0.2)
% ylabel('Time')
% title('compare time of algorithms')

%% noise
% 
% final_T_noise =[3.0739    3.1049   16.1083    12.8815];
% final_R_noise =[0.4182    0.5422   0.8745    0.9523];
% final_MSE_noise =[0.6903 0.6375 0.3566 0.2267];


% final_T_noise =[3.0739    3.1049   16.1083    12.8815];
% final_R_noise =[0.4182    0.5422   0.8745    0.9523];
% final_MSE_noise =[0.6903 0.6375 0.3566 0.2267];
% X = categorical({'gd','gda','bfg','lm'});
% X = reordercats(X,{'gd','gda','bfg','lm'});

final_T_noise =[129.7709 245.4535];
final_R_noise =[0.9999   0.9841];
final_MSE_noise =[0.0074 0.1232];
X = categorical({'lm','br'});
X = reordercats(X,{'lm','br'});



Y_T = [final_T;final_T_noise]';
Y_R = [final_R;final_R_noise]';
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



