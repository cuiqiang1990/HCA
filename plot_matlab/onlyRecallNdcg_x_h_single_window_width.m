
% =========================================================================
% :  Dotted line,       点
% -. Dash-dotted line,  点、短杠
% -- Dashed line,       短杠
% -  Solid line,        实线

% =========================================================================
% 数据准备。top-k画成折线图，auc画成几个点。
clear;
clc;
load x_h_single_data              
disp('Loading: onlyRecallNdcg_x_h_single_data.mat');
lines_x = {'<-', 'p-', 'd-', 'o-'};
lines_h = {'*-', 's-', 'x-', '^-'};
colors_x = {[1 0.2 0],  [1 0 1],    [0 0 0.5],   [0 0.5 1]};
colors_h = {[0 0.5 0],  [0 0 1],    [0, 0.5, 1], [1 0 0]};
sub = [141, 142, 143, 144];
a = 't';               % 'taobao'拼写正确则为运行t3数据库，否则运行a1数据库
if strcmp(a, 't3')
    dataset = t3;
    ylims = {       % y轴顶上留出空来，不然legend会压住刻度值
    [0.5, 1.59], [1.2, 1.79], ...
    [0.6, 1.59], [1.48, 1.79]};
else
    dataset = o1;
    ylims = {       
    [10, 28], [8, 17.9], ...
    [10, 26], [8.5, 17.9]};    
end

figure();
set(gca,'FontSize',15);
x = data_top_k;
set(gca, 'XTick', x);   % 指定x轴刻度标识

% =========================================================================
% 画折线子图：181, 182, 183，184。对应方法：'x2', 'x3', 'x4', 'x5'
lines = lines_x;
colors = colors_x;
for num = [1, 2]
    name = data_evaluation{num};    % recall,map,ndcg,auc
    data = dataset{num};
    subplot(sub(num));
    for i = [1, 2, 3, 4]      
        plot(x, data{i}, lines{i}, 'Color',colors{i}, 'LineWidth', 1, 'MarkerSize', 7);
        hold on;        
    end
    xlabel('top-k');
    labels = {'5', '10', '15'};  
    set(gca, 'XTick', x, 'XTickLabel', x);   % 指定x轴显示标识  
    xlim([4 16])
    ylabel(name)
    ylim(ylims{num})
end
leg = data_method_x;
hl = legend(leg);
set(hl, 'Location', 'NorthOutside', 'Orientation', 'horizontal', 'Box', 'off');

% =========================================================================
% 画折线子图：185, 186, 187，188。对应方法：'h2', 'h3', 'h4', 'h5'
lines = lines_h;
colors = colors_h;
for num = [3, 4]
    name = data_evaluation{num};    % recall,map,ndcg,auc
    data = dataset{num};
    subplot(sub(num));
    for i = [1, 2, 3, 4]      
        plot(x, data{i}, lines{i}, 'Color',colors{i}, 'LineWidth', 1, 'MarkerSize', 7);
        hold on;       
    end
    xlabel('top-k');
    labels = {'5', '10', '15'};  
    set(gca, 'XTick', x, 'XTickLabel', x);   
    xlim([4 16])
    ylabel(name)
    ylim(ylims{num})
end 
leg = data_method_h;
hl = legend(leg);
set(hl, 'Location', 'NorthOutside', 'Orientation', 'horizontal', 'Box', 'off');
















