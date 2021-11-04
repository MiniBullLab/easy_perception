#ifndef _NET_PROCESS_H_
#define _NET_PROCESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <glog/logging.h>
#include <glog/raw_logging.h>


class NetProcess
{
public:
    NetProcess();
    ~NetProcess();

    int init_network();
    int send_result(const std::string &lpr_result, const int code);

private:
    int udp_socket_fd;
    int upd_port;

    int dest_port;
	struct sockaddr_in dest_addr;
};

#endif // _NET_PROCESS_H_