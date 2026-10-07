# Ultra 算法组特招考核提交材料

本仓库保存 ROS 2 学习笔记、必做与选做二核心代码及必要的验证证据。

| 内容 | 位置 |
|---|---|
| 考核要求 | [特招考核任务.docx](特招考核任务.docx) |
| ROS 2 学习笔记 | [ROS2-李浩.pdf](ROS2-李浩.pdf) |
| 必做图像处理链路 | [armor_detector.cpp](核心代码/src/armor_detector.cpp) |
| 选做二核心算法与入口 | [optional_armor.cpp](核心代码/src/optional_armor.cpp)、[optional_main.cpp](核心代码/src/optional_main.cpp) |
| 构建及原工程提交记录 | [编译与Git记录.txt](核心代码/编译与Git记录.txt) |
| 截图、视频、日志与量化数据 | [证据](证据) |

选做二三组各 360 帧，最大距离误差分别为 0.03215、0.03318、0.02617 m，最长连续丢失为 0、0、1 帧。结果适用于蓝灯、号码 3、固定相机的受控仿真；按每帧获得一个可见目标统计，允许换板。全部可见板的召回率分别为 100%、68.0%、67.4%。详见[量化结果](证据/选做二/量化结果.json)和[逐帧核对](证据/选做二/逐帧核对.csv)。

代码已于 2026-10-07 整理可读性，全新编译无警告、无错误，1080 帧检测数据与重构前逐字节一致。原工程来源提交为 `d35ff56dd85fff610bcfab7e564b7040af1e1cd4`；本仓库为提交材料的独立快照。

在安装 OpenCV 开发库的 Ubuntu 环境中构建：

```bash
cmake -S "核心代码" -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

同济项目日志仍有 `Target diverged!`，模型收敛与可视化准确性尚待补证。PDF 原样保留，第 2 页应纠正：Discovery 负责发现通信实体并匹配连接，实际消息传输由中间件完成。
