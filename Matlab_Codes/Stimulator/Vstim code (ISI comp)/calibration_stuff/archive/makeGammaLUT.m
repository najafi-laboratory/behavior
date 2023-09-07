
%First read text file of luminance values read in from the spectrometer thingy:

root = '/Matlab_code/calibration_stuff/measurements/';

%f = 'SGICRT/luminance';
%f = 'LCD/luminance';

%f = 'LCD 5-3-10 PR601/luminance';
f = 'CRT 5-2-10 PR601/luminance';

fid = fopen([root f],'r');

L = [];

l = fgetl(fid);

while(l ~= -1)
    s = sscanf(l,'%f,%f,%f,%f,%f');
    
    if round(s(1)) == -4996 | round(s(1)) == 10  %The 'or' needs to be a single bar
        s = s(1)*ones(5,1);
    end
    L = [L;s(3)];  %luminance is the third element
    l = fgetl(fid);
    %l = fgetl(fid); %if there is a space between them
end
L = reshape(L,length(L(:,1))/3,3);  %luminance


domall = 0:255;
dom = 0:255/(length(L(:,1))-1):255;
clear gam
for i = 1:3
    
    id = find((L(:,i)) ~= -4996 & (L(:,i)) ~= 10);
    
    domdum{i} = dom(id);
    Ldum{i} = L(id,i);
    
    if L(1,i) == 10

        domdum{i} = [0; domdum{i}(:)];
        Ldum{i} = [0; Ldum{i}(:)];
        
        Lhat(:,i) = interp1(domdum{i},Ldum{i},domall);
        
    end

    
end


figure,
for i = 1:3
    plot(domdum{i},Ldum{i})
    hold on
end
hold on, plot(domall,Lhat,'k')

%Convert to Look-up-table
Lhat = Lhat./(ones(256,1)*Lhat(end,:));

buf = linspace(0,1,256)'; 
gammaLUT = linspace(0,1,256)'; 
for i = 1:3
    bufLUT(:,i) = interp1(Lhat(:,i),buf,gammaLUT);
end
figure,plot(bufLUT)

%Save raw values and look up table

L = Lhat;
%save([root f],'bufLUT','L')

