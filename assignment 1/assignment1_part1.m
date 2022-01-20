x = linspace(0,1,21); % creates 21 datapoints uniformily distributed in the interval [0,1].
y = -sin(.8*pi*x); % computes the image of these 21 datapoints in a new vector y.
plot(x,y)
xlabel('x')
ylabel("y")
title('y = sin(x)')

% net = fitnet(1);
% net.layers{1}.transferFcn = 'purelin';
% net = configure(net,x,y);
% net.inputs{1}.processFcns = {};
% net.outputs{2}.processFcns = {};
% [net, tr] = train(net,x,y);

net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net,x,y);


