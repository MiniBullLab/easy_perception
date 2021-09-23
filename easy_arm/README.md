easy_arm
=================================================

# docker环境下编译
1. 启动workspace镜像
2. 在easy_data目录下
    ```
    git clone https://github.com/MiniBullLab/easy_perception.git
    ```
3. 下载amba sdk
4. 进入在workspace镜像中
5. cd /easy_data/easy_perception/easy_arm/ambarella
6. ./amba_build.sh aarch64
7. 运行先需要运行
    ```
    modprobe cavalry && /usr/local/bin/cavalry_load -f /lib/firmware/cavalry.bin -r
    ```
    然后运行相应的应用程序