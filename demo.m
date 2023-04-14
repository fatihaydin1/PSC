% DEMO
clc;
clear;

load BanknoteAuthentication;
[ X, Y ] = divideTable( BanknoteAuthentication );
X = normalize(X,'range');

rpt = 5;
ACC = zeros(1, rpt);
T = zeros(1, rpt);

options = optimoptions('particleswarm');
options.SelfAdjustmentWeight = 1.5;
options.SocialAdjustmentWeight = 1.9;
options.MinNeighborsFraction = 0.25;
options.SwarmSize=10;

for j = 1 : rpt
    disp(j);
    [ACC(j), T(j)] = run(X, Y, options);
end
results = [mean(ACC)    std(ACC)     mean(T)      std(T)];

clearvars -except results options ...
    BanknoteAuthentication BloodTransfusion BreastCancer ClimateModel ConnectionistBench...
    DiabeticRetinopathy eegeyestate Haberman HTRU2 Ionosphere Madelon mozilla4 ...
    ParkinsonSpeech QSARBiodegradation VertebralColumn;


%%
function [ accuracy, T ] = run( X, Y, options )
    
    T = 0;
    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
           
    for i = 1:10
        test = (indices == i);
        train = ~test;
                
        trainY = Y(train,:);
        trainX = X(train,:);
        testX = X(test,:);
        
        tic;
        Mdl = psc(trainX, trainY, options);
        T = T + toc;
        p = psc(testX, Mdl);
        p(p==1) = 2;
        p(p==-1) = 1;
        [~, ~, GL] = grp2idx(Y);
        predictions(test, 2) = GL(p);
    end
    
    accuracy = sum(predictions(:,1) == predictions(:,2))*100/length(Y);
    T = T / 10;
end


%% Separate the dataset into the input matrix and the output vector
function [ X, Y ] = divideTable( DATASET )

    if istable(DATASET)
        X = table2array(DATASET(:,1:end-1));        
        Y = categorical(DATASET.Class);
    else
        error('The parameter must be a table, not a %s.', class(DATASET));
    end
end
