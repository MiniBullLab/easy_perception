# amba_arm

# 编译
1. 安装amba当前最新交叉编译工具
2. 下载amba sdk, 并编译(可选)：
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
3. 设置ambarella/cmake/ambacv25_3_dependencies.cmake中sdk路径
4. cd ./ambarella
5. ./amba_build.sh aarch64
6. 在build目录中会有相应的可执行程序