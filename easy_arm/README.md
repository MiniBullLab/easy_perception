easy_arm
=================================================

# docker环境下编译
1. 启动workspace镜像
2. 在easy_data目录下
    ```
    git clone https://github.com/MiniBullLab/easy_perception.git
    ```
3. 下载amba sdk, 放在easy_data目录下, 并编译(可选)：
    ```
    sudo apt-get install git-core gnupg flex bison gperf build-essential zip curl zlib1g-dev wget mtd-utils fakeroot cramfsprogs genext2fs gawk subversion git-gui gitk unixodbc texinfo gcc-multilib g++-multilib

    sudo apt-get install autoconf libsqlite3-dev graphviz libc6-i386

    cd /easy_data/cv22_linux_sdk/ambarella/

    source build/env/aarch64-linaro-gcc.env

    cd boards/cv22_walnut

    ls -al config/

    make sync_build_mkcfg

    make defconfig_public_linux

    make -j8

    ```
4. 进入在workspace镜像中
5. cd /easy_data/easy_perception/easy_arm/ambarella
6. ./amba_build.sh aarch64
7. 运行先需要运行
    ```
    modprobe cavalry && /usr/local/bin/cavalry_load -f /lib/firmware/cavalry.bin -r
    ```
    然后运行相应的应用程序