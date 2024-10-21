# README

##  Code language and libraries 

The Radar Simulation System is based on **MATLAB**, using  ***Phased Array System Toolbox***.

The introduction of the [Phased Array System Toolbox]([Phased Array System Toolbox Documentation - MathWorks 中国](https://ww2.mathworks.cn/help/phased/index.html?s_tid=CRUX_lftnav))



## Code

```matlab
% --------------------- 初始化参数 --------------------- 
frequency = 10e9; % 雷达工作频率 10 GHz
lambda = physconst('LightSpeed') / frequency; % 波长
pulseWidth = 1e-6; % 脉冲宽度
prf = 1e3; % 脉冲重复频率 (PRF)
maxRange = 2000; % 最大探测距离
rangeRes = 1; % 距离分辨率
RCS = 0.1; % 无人机群的RCS假设为0.1平方米
scanRate = 360 / 3; % 雷达旋转速度, 每3秒转360°
numDrones = 10; % 无人机数量
formationSpacing = 10; % 编队间距（米）
```

​	设置了雷达工作频率（10 GHz）、脉冲宽度（1微秒）、脉冲重复频率、最大探测距离（2000米）和距离分辨率。

无人机数量设置为10，定义了初始位置和巡航速度，使无人机在x和y方向随机移动。

```matlab
% --------------------- 定义编队形态 --------------------- 
formationType = 'rectangle'; % 编队类型，可以选择'line', 'rectangle', 'triangle', 'diamond', 'V'

% 根据编队类型生成无人机初始位置
if strcmp(formationType, 'line')
    droneInitialPositions = [linspace(0, (numDrones-1) * formationSpacing, numDrones); ...
                             zeros(1, numDrones); ...
                             50 * ones(1, numDrones)]; % 无人机高度固定为50米
elseif strcmp(formationType, 'rectangle')
    rows = ceil(sqrt(numDrones)); % 计算行数
    cols = ceil(numDrones / rows); % 计算列数
    [X, Y] = meshgrid(0:formationSpacing:(cols-1) * formationSpacing, 0:formationSpacing:(rows-1) * formationSpacing);
    droneInitialPositions = [X(:)'; Y(:)'; 50 * ones(1, numel(X(:)))]; % 维度一致
elseif strcmp(formationType, 'triangle')
    droneInitialPositions = zeros(3, numDrones);
    for i = 1:numDrones
        row = floor(sqrt(i)); % 当前行
        col = i - (row * (row + 1)) / 2; % 当前列
        droneInitialPositions(1, i) = col * formationSpacing; % x坐标
        droneInitialPositions(2, i) = row * formationSpacing; % y坐标
        droneInitialPositions(3, i) = 50; % z坐标
    end
elseif strcmp(formationType, 'diamond')
    droneInitialPositions = zeros(3, numDrones);
    for i = 1:numDrones
        row = floor(sqrt(i)); % 当前行
        offset = mod(i, (row + 1)); % 当前偏移
        droneInitialPositions(1, i) = (row - offset) * formationSpacing; % x坐标
        droneInitialPositions(2, i) = (row + offset) * formationSpacing; % y坐标
        droneInitialPositions(3, i) = 50; % z坐标
    end
elseif strcmp(formationType, 'V')
    droneInitialPositions = zeros(3, numDrones);
    for i = 1:numDrones
        angle = linspace(-pi/4, pi/4, numDrones); % V字形的角度
        droneInitialPositions(1, i) = (i - (numDrones / 2)) * formationSpacing * cos(angle(i)); % x坐标
        droneInitialPositions(2, i) = (i - (numDrones / 2)) * formationSpacing * sin(angle(i)); % y坐标
        droneInitialPositions(3, i) = 50; % z坐标
    end
else
    error('Unsupported formation type');
end

% 定义无人机的巡航速度 (vx, vy, vz)，单位为米/秒
droneVelocities = [20 * randn(numDrones, 1), ...
                   20 * randn(numDrones, 1), ...
                   zeros(numDrones, 1)]; % 无人机沿x, y方向巡航，不在z方向移动

```

这部分代码给出了五类无人机编队模式以及分布计算方法，其中z方向高度不发生改变，无人机进行水平飞行。

​	同时通过**randn（）**随机数随机定义无人机巡航速度。

```matlab
% --------------------- 创建雷达平台 --------------------- 
radarPlatform = phased.Platform('InitialPosition', [0; 0; 10], ...
                                'Velocity', [0; 0; 0]);

% --------------------- 创建雷达和目标 --------------------- 
transmitter = phased.Transmitter('PeakPower', 1e3, ...
                                 'Gain', 40);
receiver = phased.ReceiverPreamp('Gain', 40, ...
                                 'NoiseFigure', 2);

dronesRCS = RCS * ones(1, numDrones); % 每个无人机的RCS
droneTargets = phased.RadarTarget('MeanRCS', dronesRCS, ...
                                  'OperatingFrequency', frequency);
```

​	这是对雷达平台以及目标的定义部分，确定了雷达的坐标，接发天线参数。同时将无人机RSC分配给各个无人机，并定义无人机目标

```matlab
rotateRound = 10; % 定义扫描角度圈数
azimuthAngles = 0:scanRate:360 * rotateRound;

% 创建结果表格
resultTable = table([], [], [], [], [], [], [], ...
                    'VariableNames', {'Time', 'SlantRange', 'RadialVelocity', ...
                                      'AzimuthAngle', 'ElevationAngle', 'Round', 'ClutterLabel'});

% 创建集群结果表格
clusterResultTable = table([], [], [], [], [], [], [], ...
                            'VariableNames', {'Time', 'ClusterCenterX', 'ClusterCenterY', 'ClusterCenterZ', ...
                                              'AverageVelocityX', 'AverageVelocityY', 'AverageVelocityZ'});

% 计算每个方位角更新的时间步长
rotationTimePerStep = 3 / (360 / scanRate); % 每个方位角更新的时间 (秒)
```

​	在仿真雷达扫描之前，以工作圈数的方式设定了雷达的工作范围。

​	为了提取仿真数据集，分别建立两个表格，用来存储雷达探测结果仿真数据和无人机编队特征数据

​	由于时间步长在初期易与脉冲重复频率（PRF）冲突，在此通过扫描速率（scanRate）单独定义旋转时间步长，保证每次输出的间隔为1s。

```matlab
% 模拟雷达扫描
simTime = 0; % 初始时间
for az = azimuthAngles
    simTime = simTime + rotationTimePerStep;

    % 更新无人机位置
    for i = 1:numDrones
        droneInitialPositions(:, i) = droneInitialPositions(:, i) + droneVelocities(i, :)' * rotationTimePerStep; % 更新位置
    end

    % 计算集群中心和速度
    clusterCenter = mean(droneInitialPositions, 2); % 计算集群中心
    averageVelocity = mean(droneVelocities, 1)'; % 计算集群速度，转置为列向量

    % 添加集群数据行
    clusterRow = {simTime, clusterCenter(1), clusterCenter(2), clusterCenter(3), ...
                  averageVelocity(1), averageVelocity(2), averageVelocity(3)}; 
    clusterResultTable = [clusterResultTable; clusterRow]; % 追加新行

    % 记录每个无人机数据
    for i = 1:numDrones
        [range, angle] = rangeangle(droneInitialPositions(:, i), radarPlatform.InitialPosition);
        radialVelocity = radialspeed(droneInitialPositions(:, i), droneVelocities(i, :)', radarPlatform.InitialPosition, [0; 0; 0]);
        azimuthAngle = angle; % 方位角
        elevationAngle = atan2(droneInitialPositions(3, i) - radarPlatform.InitialPosition(3), range); % 俯仰角

        % 添加无人机数据行
        newRow = {simTime, range, radialVelocity, azimuthAngle, elevationAngle, floor(az/360) + 1, 1};
        resultTable = [resultTable; newRow]; % 追加新行
    end

    % 生成杂波（云层、地面反射等）
    numClutter = randi([1, 5]); % 随机生成1到5个杂波
    for j = 1:numClutter
        clutterPos = [rand() * 2000; rand() * 2000; 0]; % 随机生成杂波位置
        [clutterRange, clutterAngle] = rangeangle(clutterPos, radarPlatform.InitialPosition);
        clutterElevationAngle = atan2(clutterPos(3) - radarPlatform.InitialPosition(3), clutterRange);

        % 随机生成杂波的径向速度
        clutterRadialVelocity = randi([-30.000, 30.000]); 

        % 添加杂波数据行
        clutterRow = {simTime, clutterRange, clutterRadialVelocity, clutterAngle, clutterElevationAngle, floor(az/360) + 1, 0};
        clutterRowTable = cell2table(clutterRow, 'VariableNames', resultTable.Properties.VariableNames);
        resultTable = [resultTable; clutterRowTable]; % 追加杂波数据行
    end
end
```

​	该部分为雷达扫描仿真部分，通过 `rangeangle` 计算目标相对于雷达的位置，并用 `radialspeed` 计算径向速度。

每个扫描时段，随机生成一些杂波目标（数量为1到5个），并赋予它们随机的方位角和径向速度以模拟实际的杂波干扰。

​	为提供标签区分无人机与杂波，将无人机和杂波分别添加到输出数据中。

​	在此过程假设雷达对于无人机的扫描是准确的。

```
% 输出仿真结果
disp(resultTable);
disp(clusterResultTable);

% 将表格数据保存到文件
writetable(resultTable, 'RadarSimulationResults.csv');
writetable(clusterResultTable, 'ClusterResults.csv');
```

​	结果输出部分，将仿真结果保存为csv格式文件
