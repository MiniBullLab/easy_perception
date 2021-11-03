#ifndef _NET_PROCESS_H_
#define _NET_PROCESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>


class NetProcess
{
public:
    NetProcess();
    ~NetProcess();

private:
    int udp_socket_fd;
    int upd_port;
};

#endif // _NET_PROCESS_H_