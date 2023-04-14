%% Particle Swarm Classifier (PSC)
function [ varargout ] = psc( X, varargin )

    if nargin == 3
        Y = varargin{1};
        options = varargin{2};
        
        [y, ~] = grp2idx(Y);
        y(y==1) = -1;
        y(y==2) = 1;

        [row, col] = size(X);
        X = [ones(row, 1) X];

        rng default
        fun = @(w)objectiveFunc(w, X, y);
        [Mdl.w, Mdl.fval, Mdl.exitflag, Mdl.output] = particleswarm(fun,col + 1, [], [],options);
        
        varargout{1} = Mdl;
    elseif nargin == 2
        Mdl = varargin{1};
        
        X = [ones(size(X, 1), 1) X];
        p = sign(X*Mdl.w');
        p(p==0) = -1;
        
        varargout{1} = p;
    end
end


%%
function [ cost ] = objectiveFunc( w, X, y )

    row = size(X, 1);
    
    p = sign(X*w');
    p(p==0) = -1;
    
    cost = sum(p~=y) * 100 / row;
end

