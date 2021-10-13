function [BestCost,BestValue]=CCLNNA(fhd,nPop,nVar,VarMin,VarMax,MaxIt,X)

% Last Revised: 12th August 2018
%(CostFunction,nPop,nVar,VarMin,VarMax,MaxIt,X,T)
% Neural Network Algorithm (NNA) (Standard Version)

% This code is prepared for single objective function (minimization), unconstrained, and continuous optimization problems.
% Note that in order to obey the copy-right rules, please cite our published paper properly. Thank you.

% Ali Sadollah, Hassan Sayyaadi, Anupam Yadav (2018) “A dynamic metaheuristic optimization model inspired by biological nervous systems: Neural network algorithm? Applied Soft Computing, 71, pp. 747-782.

% INPUTS:

% objective_function:           Objective function which you wish to minimize or maximize
% LB:                           Lower bound of a problem
% UB:                           Upper bound of a problem
% nvars:                        Number of design variables
% npop                          Population size
% max_it:                       Maximum number of iterations

% OUTPUTS:

% Xoptimum:                     Global optimum solution
% Objective:                    Cost of global optimum solution
% NFEs:                         Number of function evaluations
% FMIN:                         Cost reduction history
% Elapsed_Time:                 Elasped time for solving an optimization problem

%% ==========================================================================
% Default values for the NNA
%==========================================================================
%% --------------------Initialization----------------------------------------
X_LB=repmat(VarMin,nPop,1);
X_UB=repmat(VarMax,nPop,1);
beta=1;
x_pattern=zeros(nPop,nVar);
cost=zeros(nPop,1);
% Creat random initial population
for i=1:nPop
    x_pattern(i,:)=X(i,:);
    cost(i)=fhd(x_pattern(i,:));
end

[COST,index]=min(cost);
%--------------------------------------------------------------------------
% Creat random initial weights with constraint of Summation each column = 1
ww=ones(1,nPop)*0.5;
w=diag(ww);
for i=1:nPop
    t=rand(1,nPop-1)*0.5;
    t=(t./sum(t))*0.5;
    w(w(:,i)==0,i)=t;
end
%--------------------------------------------------------------------------
% w=rand(npop,npop);       %% An alternative way of initializing weights
% for i=1:npop
%     w(:,i)=w(:,i)./sum(w(:,i));       % Summation of each column = 1
% end
%--------------------------------------------------------------------------
XTarget=x_pattern(index,:);   % Best obtained solution
Target=COST;                  % Best obtained objetive function value
wtarget=w(:,index);           % Best obtained weight (weight target)

%% -------------------- Main Loop for the NNA -------------------------------
FMIN=zeros(MaxIt,1);
tic
BestCost(1)=COST;
mm=rand; %% Intilize logitic map
for ii=2:MaxIt
    mm=4*mm*(1-mm); % Eq. (15)
    IND=randperm(nPop);
    IND1=IND(1:nPop/2);
    IND2=IND(nPop/2+1:nPop);
    for i=1:nPop/2
        if cost(IND1(i))<cost(IND2(i))
            WINNER(i)=IND1(i);
            LOSER(i)=IND2(i);
        else
            LOSER(i)=IND1(i);
            WINNER(i)=IND2(i);
        end
    end
    MW=(mean(w))';
    MM=mean(x_pattern);
    %------------------ Creating new solutions ----------------------------
    x_new=w*x_pattern;
    x_pattern=x_new+x_pattern;
    %------------------- Updating the weights -----------------------------
    for i=1:nPop
        w(:,i)=abs(w(:,i)+((wtarget-w(:,i))*2.*rand(nPop,1))+beta.*2.*(MW-w(:,i)).*rand(nPop,1));
    end
    
    for i=1:nPop
        w(:,i)=w(:,i)./sum(w(:,i));    % Summation of each column = 1
    end
    
    %----------------------- Creat new input solutions --------------------
    for i=1:nPop/2
        if rand<mm
            N_Rotate=ceil(beta*nVar);
            
            for j=1:nVar
                xx=VarMin+(VarMax-VarMin).*rand(1,nVar);
                fi=rand;
                gi=1-fi;
                if rand<rand %Eq. (12)
                    xx(j)=fi*x_pattern(WINNER(i),j)+gi*xx(j);
                else
                    xx(j)=fi*XTarget(j)+gi*xx(j);
                end
            end
            rotate_postion=randperm(nVar);rotate_postion=rotate_postion(1:N_Rotate);
            
            for m=1:N_Rotate
                x_pattern(LOSER(i),rotate_postion(m))=xx(m);
            end
        else
            x_pattern(LOSER(i),:)=x_pattern(LOSER(i),:)+(XTarget-x_pattern(LOSER(i),:))*2.*rand(1,nVar)+2.*(MM-x_pattern(LOSER(i),:)).*rand(1,nVar)*beta;%Eq. (14)
        end
    end
    for i=1:nPop
        if rand<mm
            %---------- Bias for weights ----------------------------------
            N_wRotate=ceil(beta*nPop);
            
            w_new=rand(N_wRotate,nPop);
            rotate_position=randperm(nPop);rotate_position=rotate_position(1:N_wRotate);
            
            for j=1:N_wRotate
                w(rotate_position(j),:)=w_new(j,:);
            end
            
            for iii=1:nPop
                w(:,iii)=w(:,iii)./sum(w(:,iii));   % Summation of each column = 1
            end
        end
    end
    for i=1:nPop/2
        %------------ Transfer Function Operator
        %----------------------zz
            x_pattern(WINNER(i),:)=x_pattern(WINNER(i),:)+(XTarget-x_pattern(WINNER(i),:))*2.*rand(1,nVar)+2.*(MM-x_pattern(WINNER(i),:)).*rand(1,nVar)*beta; %Eq. (10)
    end
    
    % ---------------------- Bias Reduction -------------------------------
    beta=beta*0.99;

    %----------------------------------------------------------------------
    x_pattern=max(x_pattern,X_LB);    x_pattern=min(x_pattern,X_UB);     % Check the side constraints
    %-------------- Calculating objective function values -----------------
    for i=1:nPop
        cost(i)=fhd(x_pattern(i,:));
    end
    
    %% ------ Selection ---------------------------------------------------
    [FF,Index]=min(cost);
    
    if FF<Target
        Target=FF;
        XTarget=x_pattern(Index,:);
        wtarget=w(:,Index);
    else
        [~,Indexx]=max(cost);
        x_pattern(Indexx,:)=XTarget;
        w(:,Indexx)=wtarget;
    end
    BestCost(ii)=Target;
    %% Display
    %   disp(['Iteration: ',num2str(ii),'   Objective= ',num2str(Target,15)]);
    FMIN(ii)=Target;
    
    
    
end

%% -------------------------------- NNA Finishes ----------------------------
BestValue=Target;
end