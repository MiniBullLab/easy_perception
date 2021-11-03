#include "net_process.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>

volatile int net_run_flag = 1;

static void * upd_recv_msg(void *arg)
{
	int ret = 0;
	int *socket_fd = (int *)arg;//通信的socket
	struct sockaddr_in  src_addr = {0};  //用来存放对方(信息的发送方)的IP地址信息
	int len = sizeof(src_addr);	//地址信息的大小
	char msg[1024] = {0};//消息缓冲区
	while(net_run_flag > 0)
	{
		ret = recvfrom(*socket_fd, msg, sizeof(msg), 0, (struct sockaddr *)&src_addr, (socklen_t*)len);
		if(ret > 0)
		{
			printf("[%s:%d]",inet_ntoa(src_addr.sin_addr),ntohs(src_addr.sin_port));
			printf("msg=%s\n",msg);
			if(strcmp(msg, "exit") == 0 || strcmp(msg, "") == 0)
			{
				net_run_flag = 0;
				break;
			}
			memset(msg, 0, sizeof(msg));//清空存留消息	
		}
	}
	//关闭通信socket
	close(*socket_fd);
	std::cout << "upd recv msg thread quit" << std::endl;
	return NULL;
}

static void* upd_broadcast_send(void* save_data)
{
	int rval = 0;
	int broadcast_port = 8888;
	int on = 1; //开启
	struct sockaddr_in broadcast_addr = {0};
	char buf[1024] = "LPR Runing!";
	int broadcast_socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (broadcast_socket_fd == -1)
    {
        printf("create socket failed ! error message :%s\n", strerror(errno));
        return NULL;
    }
	//开启发送广播数据功能
	rval = setsockopt(broadcast_socket_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
	if(rval < 0)
	{
		perror("setsockopt fail\n");
		return NULL;
	}
	//设置当前网段的广播地址 
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(broadcast_port);
    broadcast_addr.sin_addr.s_addr = inet_addr("10.0.0.255");  //设置为广播地址
	while(net_run_flag > 0)
	{
		std::cout << "heart loop!" << std::endl;
		sendto(broadcast_socket_fd, buf, strlen(buf), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
		sleep(1);
	}
	strcpy(buf, "LPR Stop!");
	sendto(broadcast_socket_fd, buf, strlen(buf), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
	close(broadcast_socket_fd);
	std::cout << "upd broadcast thread quit" << std::endl;
	return NULL;
}

NetProcess::NetProcess()
{

}

NetProcess::~NetProcess()
{
	
}