%--------------------------------Method 1--------------------------------%
%For each subject, use the first five images (1.pgm to 5.pgm) for training 
%the subspace. Use the files 6.pgm to 10.pgm for the performance evaluation

%Create filepaths for all subjects
filepaths = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s',a,'\');
    filepaths{i}=x
end
%Create directories (string) of all subject folders
sub_dir = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s',a,'\*.pgm');
    x = char(x)
    x = dir(x)
    sub_dir{i}=x
end

%Create a training set made of the first 5 images for each test subject
%(200 images total)
training = cell(1,200);
%Create a training set made of the first 5 images for each test subject
%(200 images total)
testing = cell(1,200);

%Read all images into the appropriate training and testing arrays
[training,testing] = sub_images(sub_dir,filepaths)

%Convert cell arrays to matrices
training = cell2mat(training);
testing = cell2mat(testing);

%---------------------------------LDA---------------------------------%
%Determine the overall mean
overall_mean = mean(training,2)
M = repmat(overall_mean,[1,200])
%Determine the mean of each class
j=1
for i = 0:5:195
     class_mean(:,j)=mean(training(:,i+1:i+5),2);
     j = j+1;
end
sw = 0;
a = 1;
b = 5;
data = [];

%Find the within class scatter matrix by subtracting the mean of each class
%from the data of each class
for i=1:40
    temp = a
    for x=1:5
        data(:,x) = training(:,temp);
        temp = temp+1
    end
    d = data - class_mean(:,i);
    co = d*d';
    sw = sw + co;
    a = a + 5;
    b = b + 5;
end
invsw = pinv(sw);
a = 1;
b = 1;
sb = 0;
temp = [];
%Determine the between class scatter matrix by subtracting the overall 
%mean from each class mean
for i=1:40
    temp = (class_mean(:,i)-overall_mean)*(class_mean(:,i)-overall_mean)';
    sb = sb + temp
    a = a + 5;
    b = b + 5;
end

v = cov((training-M)');
%PCA eigenvalues and eigenvectors
[PCA_Vecs,PCA_Vals] = eig(v);
%PCA Eigenspace selection
PCA_eigenspace = PCA_Vecs(:,(10304-159:10304));
%Project within class scatter on PCA subspace
sw_projection = PCA_eigenspace'*sw*PCA_eigenspace;
%Project between class scatter on PCA subspace
sb_projection = PCA_eigenspace'*sb*PCA_eigenspace;
%Eigenvectors and eigenvalues
[eigenvectors,eigenvalues] = eig(sb_projection,sw_projection);
training_projection = PCA_eigenspace*eigenvectors;
%LDA Eigenspace selection
training_projection_eigensel = training_projection(:,1:40);
%Training projection on Eigenspace
training_projection_eigenspace = training_projection_eigensel'*(training-M);

%%%Testing
%Mean of the test data
m_test = mean(testing,2);
M_test = repmat(m_test,[1,200]);

%Testing projecting on Eigenspace
testing_projection = training_projection_eigensel'*(testing-M_test);

%Euclidean distance 
D = pdist2(training_projection_eigenspace',testing_projection','Euclidean');

%Labels (determining what was correctly classified and what was not)
results = zeros(200,200);
for i = 1: 200
    for k = 1: 200
        if(fix((i-1)/10)==fix((k-1)/10))
            results(i,k)=0
        else
            results(i,k)=1;
        end
    end
end

%Find and plot the ROC curve
ezroc3(D,results,2,'',1);

%Function to import images from att_faces folder
function [train,test] = sub_images(sub_directory,filepath)
%200 training images (first 5 from each subject)
train = cell(1,200);
%200 testing images (last 5 from each test subject)
test = cell(1,200);
%a and b are placeholders for traversing through the train and test cell
%arrays
a = 1;
b = 1;

%Loop for all 40 subjects
for k = 1: 40
    fp = filepath{k}
    x = sub_directory{k}
    directory = x
    for i = 1: 10
        if i < 6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Read image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to training set
            train{a}=temp;
            a = a+1;
        end
        if i>=6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Raed image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to testing set
            test{b}=temp;
            b = b+1;
        end
    end
end
end

%Function plotting the ROC curve
function [roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(H,T,plot_stat,headding,printInfo)%,rbst
t1=min(min(min(H)));
t2=max(max(max(H)));
num_subj=size(H,1);

stp=(t2-t1)/500;   %step size here is 0.2% of threshold span, can be adjusted

if stp==0   %if all inputs are the same...
    stp=0.01;   %Token value
end
ALLthr=(t1-stp):stp:(t2+stp);
if (nargin==1 || (nargin==3 &&  isempty(T))||(nargin==2 &&  isempty(T))||(nargin==4 &&  isempty(T))||(nargin==5 &&  isempty(T)))  %Using only H, multi-class case, and maybe 3D or no plot
    GAR=zeros(503,size(H,3));  %initialize for accumulation in case of multiple H (on 3rd dim of H)
    FAR=zeros(503,size(H,3));
    gen=[]; %genuine scores place holder (diagonal of H), for claculation of d'
    imp=[]; %impostor scores place holder (non-diagonal elements of H), for claculation of d'
    for setnum=1:size(H,3); %multiple H measurements (across 3rd dim, where 2D H's stack up)
        gen=[gen; diag(H(:,:,setnum))]; %digonal scores
        imp=[imp; H(find(not(eye(size(H,2)))))]; %off-diagonal scores, with off-diagonal indices being listed by find(not(eye(size(H,2)))) 
        for t=(t1-stp):stp:(t2+stp),    %Note that same threshold is used for all H's, and we increase the limits by a smidgeon to get a full curve
            ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
            id=H(:,:,setnum)>t;
            
            True_Accept=trace(id);  %TP
            False_Reject=num_subj-True_Accept;  %FN
            % In the following, id-diag(diag(id)) simply zeros out the diagonal of id
            True_Reject=sum( sum( (id-diag(diag(id)))==0 ) )-size(id,1); %TN, number of off-diag zeros. We need to subtract out the number of diagonals, as 'id-diag(diag(id))' introduces those many extra zeros into the sum
            False_Accept=sum( sum( id-diag(diag(id)) ) ); %FP, number of off-diagonal ones
            
            GAR(ind,setnum)=GAR(ind,setnum)+True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
            FAR(ind,setnum)=FAR(ind,setnum)+False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        end
    end
    GAR=sum(GAR,2)/size(H,3);   %average across multiple H's
    FAR=sum(FAR,2)/size(H,3);
elseif (nargin==2 || nargin==3 || nargin == 4 || nargin == 5),   %Regular, 1-class-vs-rest ROC, and maybe 3D or no plot
    gen=H(find(T)); %genuine scores
    imp=H(find(not(T))); %impostor scores
    for t=(t1-stp):stp:(t2+stp),    %span the limits by a smidgeon to get a full curve
        ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
        id=H>t;
        
        True_Accept=sum(and(id,T)); %TP
        False_Reject=sum(and(not(id),T));   %FN
        
        True_Reject=sum(and(not(id),not(T)));   %TN
        False_Accept=sum(and(id,not(T)));   %FP
        
        GAR2(ind)=True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
        FAR2(ind)=False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        
    end
    GAR=GAR2';
    FAR=FAR2';
end
roc=[GAR';FAR'];
FRR=1-GAR;
[e ind]=min(abs(FRR'-FAR'));    %This is Approx w/ error e. Fix by linear inerpolation of neigborhood and intersecting w/ y=x
EER=(FRR(ind)+FAR(ind))/2;
area=abs(trapz(roc(2,:),roc(1,:)));
EERthr=t1+(ind-1)*stp;%EER threshold

d=abs(mean(gen)-mean(imp))/(sqrt(0.5*(var(gen)+var(imp))));   %Decidability or d'

if (nargin==1 || nargin==2 || nargin==3 || nargin == 4 || nargin == 5)
    if plot_stat == 2
        if printInfo == 1
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area) ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
        elseif printInfo == 0
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR');
        end
    elseif plot_stat == 3
        if printInfo == 1
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area)  ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        elseif printInfo == 0
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        end     
    else
        %else it must be 0, i.e. no plot
    end
end
end