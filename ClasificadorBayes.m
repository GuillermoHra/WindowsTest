%Clasificador de Bayes

%Clasificación de números (positivos, negativos, cero)
% training = [1;0;-1;-2;4;0]; %Datos para entrenar
% target_data = ['pos'; 'cer'; 'neg'; 'neg'; 'pos'; 'cer']; %Clasificación
% test = 10*randn(10,1); %Prueba del sistema
% class = classify(test, training, target_data, 'diaglinear')

%Clasificación de rangos de números
% training = [3;5;17;19;24;27;31;38;45;48;52;56;66;69;73;78;84;88];
% target_class = [0;0;10;10;20;20;30;30;40;40;50;50;60;60;70;70;80;80];
% test = [1:3:90]';
% class = classify(test, training, target_class, 'diaglinear');
% [test class]

%Metodo de detección de cáncer con red neuronal, NeuralFitting app
load ovarian_dataset;
[x, t] = ovarian_dataset;
setdemorandstream(672880951);
net = patternnet(20);
view(net);
[net, tr] = train(net, x, t);
plotperform(tr);
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX);
testClasses = testY > 0.5;
plotconfusion(testT, testY);
