#include "net_process.h"
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
			LOG(INFO) << "IP:" << inet_ntoa(src_addr.sin_addr) << " port:" << ntohs(src_addr.sin_port);
			LOG(INFO) << "msg:" << msg;
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
	LOG(INFO) << "upd recv msg thread quit!";
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
		LOG(ERROR) << "create socket failed ! error message:" << strerror(errno);
        return NULL;
    }
	//开启发送广播数据功能
	rval = setsockopt(broadcast_socket_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
	if(rval < 0)
	{
		LOG(ERROR) << "setsockopt fail";
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
	LOG(INFO) << "upd broadcast thread quit!";
	return NULL;
}

NetProcess::NetProcess()
{
	upd_port = 9999;
	udp_socket_fd = 0;

	dest_port = 9998;
    dest_addr.sin_family = AF_INET;
	dest_addr.sin_port = htons(dest_port);
	dest_addr.sin_addr.s_addr = inet_addr("10.0.0.102");
}

NetProcess::~NetProcess()
{
	if(udp_socket_fd < 0)
	{
		close(udp_socket_fd);
	}
	LOG(INFO) << "close net!";
}

int NetProcess::init_network()
{
	int rval = 0;
	struct sockaddr_in  local_addr = {0};
	struct timeval timeout;
	udp_socket_fd = socket(AF_INET,SOCK_DGRAM,0);
	if(udp_socket_fd < 0)
	{
		LOG(ERROR) << "creat socket fail";
		rval = -1;
		net_run_flag = 0;
	}
    timeout.tv_sec = 0;//秒
    timeout.tv_usec = 100000;//微秒
    if (setsockopt(udp_socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == -1) {
		LOG(ERROR) << "setsockopt failed";
		rval = -1;
		net_run_flag = 0;
    }
	bzero(&local_addr, sizeof(local_addr));
	local_addr.sin_family  = AF_INET;
	local_addr.sin_port	= htons(upd_port);
	local_addr.sin_addr.s_addr = INADDR_ANY;
	rval = bind(udp_socket_fd,(struct sockaddr*)&local_addr,sizeof(local_addr));
	if(rval < 0)
	{
		LOG(ERROR) << "bind fail!";
		close(udp_socket_fd);
		rval = -1;
		net_run_flag = 0;
	}
	return rval;
}

int NetProcess::send_result(const std::string &lpr_result, const int code)
{
	struct timeval tv;  
    char time_str[64];
	std::stringstream send_result;
	gettimeofday(&tv, NULL); 
	strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
	send_result << time_str << "|" << code << "|" << lpr_result;
	sendto(udp_socket_fd, send_result.str().c_str(), strlen(send_result.str().c_str()), \
		0, (struct sockaddr *)&dest_addr,sizeof(dest_addr));
	LOG(INFO) << send_result.str();
	return 0;
}
