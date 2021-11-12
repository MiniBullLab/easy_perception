#include "net_process.h"
#include "utility/utils.h"
#include <iostream>

static pthread_mutex_t send_mutex;
static sem_t sem_put, sem_get;
volatile int net_run_flag = 1;
volatile int recv_code = -1;

static int parseRecv(const std::string &message)
{
	int result = -1;
	size_t pos = message.find_first_of('|', 0);
	if(pos >= 0)
    {
		std::string code_str = message.substr(0, pos);
        result = atoi(code_str.c_str());
	}
	else
	{
		LOG(ERROR) << "recv msg error:" << message;
	}
	return result;
}

static void * upd_recv_msg(void *arg)
{
	int ret = 0;
	int *socket_fd = (int *)arg;//通信的socket
	struct sockaddr_in  src_addr = {0};  //用来存放对方(信息的发送方)的IP地址信息
	int len = sizeof(src_addr);	//地址信息的大小
	char msg[1024] = {0};//消息缓冲区
	prctl(PR_SET_NAME, "upd_recv_pthread");
	while(net_run_flag > 0)
	{
		ret = recvfrom(*socket_fd, msg, sizeof(msg), 0, (struct sockaddr *)&src_addr, (socklen_t*)len);
		if(ret >= 0)
		{
			struct timespec ts;
			std::string tmp_str = msg;
			LOG(WARNING) << "IP:" << inet_ntoa(src_addr.sin_addr) << " port:" << ntohs(src_addr.sin_port);
			LOG(WARNING) << "msg:" << msg;
			clock_gettime(CLOCK_REALTIME, &ts);
			ts.tv_sec += 1;
			ret = sem_timedwait(&sem_get, &ts);
			if(ret < 0) {
				LOG(ERROR) << "sem_timewait timeout";
				continue;
			}
			recv_code = parseRecv(tmp_str);
			sem_post(&sem_put);
			memset(msg, 0, sizeof(msg));//清空存留消息	
		}
	}
	//关闭通信socket
	close(*socket_fd);
	LOG(WARNING) << "upd recv msg thread quit!";
	return NULL;
}

static void* heart_send_pthread(void* arg)
{
	int rval = 0;
	int broadcast_port = 8888;
	int on = 1; //开启
	struct sockaddr_in broadcast_addr = {0};
	char buf[64] = "0|LPR Runing!";
	int *broadcast_socket_fd = (int *)arg;//通信的socket
	prctl(PR_SET_NAME, "heart_send_pthread");
	//设置当前网段的广播地址 
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(broadcast_port);
    broadcast_addr.sin_addr.s_addr = inet_addr("10.0.0.255");  //设置为广播地址
	while(net_run_flag > 0)
	{
		struct timeval tv; 
		std::stringstream send_result;
		gettimeofday(&tv, NULL); 
		strftime(buf, sizeof(buf)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
		send_result <<  0 << "|" << buf;
		sendto(*broadcast_socket_fd, send_result.str().c_str(), strlen(send_result.str().c_str()), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
		sleep(1);
	}
	strcpy(buf, "1|LPR Stop!");
	sendto(*broadcast_socket_fd, buf, strlen(buf), 0, (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr)); 
	close(*broadcast_socket_fd);
	LOG(WARNING) << "upd broadcast thread quit!";
	return NULL;
}

NetProcess::NetProcess()
{
	upd_port = 9999;
	udp_socket_fd = -1;

	dest_port = 9998;
    dest_addr.sin_family = AF_INET;
	dest_addr.sin_port = htons(dest_port);
	dest_addr.sin_addr.s_addr = inet_addr("10.0.0.102");

	broadcast_socket_fd = -1;
    broadcast_port = 8888;

	heart_pthread_id = 0;
	recv_thread_id = 0;
}

NetProcess::~NetProcess()
{
	if(net_run_flag > 0)
	{
		stop();
	}
	if(udp_socket_fd > 0)
	{
		close(udp_socket_fd);
	}
	if(broadcast_socket_fd > 0)
	{
		close(broadcast_socket_fd);
	}
	pthread_mutex_destroy(&send_mutex);
	sem_destroy(&sem_put);
	sem_destroy(&sem_get);
	LOG(INFO) << "~NetProcess()";
}

int NetProcess::init_network()
{
	int rval = 0;
	int on = 1; //开启
	struct sockaddr_in  local_addr = {0};
	struct timeval timeout;
	udp_socket_fd = socket(AF_INET,SOCK_DGRAM, 0);
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

	broadcast_socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (broadcast_socket_fd == -1)
    {
		LOG(ERROR) << "create socket failed ! error message:" << strerror(errno);
        rval = -1;
		net_run_flag = 0;
    }
	//开启发送广播数据功能
	rval = setsockopt(broadcast_socket_fd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
	if(rval < 0)
	{
		LOG(ERROR) << "setsockopt fail";
		rval = -1;
		net_run_flag = 0;
	}
	signal(SIGPIPE, SIG_IGN);

	pthread_mutex_init(&send_mutex, NULL);
	sem_init(&sem_put, 0, 0);
	sem_init(&sem_get, 0, 0);
	LOG(WARNING) << "socket init success!";
	return rval;
}

int NetProcess::start()
{
	int rval = 0;
	net_run_flag = 1;
	rval = pthread_create(&heart_pthread_id, NULL, heart_send_pthread, (void*)&broadcast_socket_fd);
	if(rval < 0)
    {
        net_run_flag = 0;
        LOG(ERROR) << "create heart pthread fail!";
    }
    else
    {
        rval = pthread_create(&recv_thread_id, NULL, upd_recv_msg, (void*)&udp_socket_fd);
        if(rval < 0)
        {
            net_run_flag = 0;
            LOG(ERROR) << "create upd recv pthread fail!";
        }
		else
		{
			LOG(INFO) << "net pthread start success!";
		}
    }
	return rval;
}

int NetProcess::stop()
{
	int ret = 0;
	net_run_flag = 0;

    LOG(WARNING) << "stop network";

	if (heart_pthread_id > 0) {
		pthread_join(heart_pthread_id, NULL);
        heart_pthread_id = 0;
	}
    if (recv_thread_id > 0) {
		pthread_join(recv_thread_id, NULL);
        recv_thread_id = 0;
	}
	LOG(WARNING) << "stop network success";
	return ret;
}

int NetProcess::process_recv()
{
	if(sem_trywait(&sem_put) == 0)
	{
		if(recv_code == 100)
		{
			send_log_path();
		}
		sem_post(&sem_get);
	}
	return 0;
}

int NetProcess::send_result(const std::string &lpr_result, const int code)
{
	struct timeval tv;  
    char time_str[64];
	std::stringstream send_result;
	gettimeofday(&tv, NULL); 
	strftime(time_str, sizeof(time_str)-1, "%Y-%m-%d_%H:%M:%S", localtime(&tv.tv_sec)); 
	send_result <<  code << "|" << time_str << "|" << lpr_result;
	pthread_mutex_lock(&send_mutex);
	sendto(udp_socket_fd, send_result.str().c_str(), strlen(send_result.str().c_str()), \
		0, (struct sockaddr *)&dest_addr,sizeof(dest_addr));
	pthread_mutex_unlock(&send_mutex);
	LOG(WARNING) << send_result.str().c_str();
	return 0;
}

int NetProcess::send_log_path()
{
	std::vector<std::string> log_list;
	std::string log_dir = "/data/glog_file/";
	ListImages(log_dir, log_list);
	std::stringstream temp_str;
	temp_str << 9 << "|";
	for (size_t index = 0; index < log_list.size(); index++) {
        temp_str << log_dir << log_list[index] << "|";
		std::cout << log_list[index] << std::endl;
	}
	pthread_mutex_lock(&send_mutex);
	sendto(udp_socket_fd, temp_str.str().c_str(), strlen(temp_str.str().c_str()), \
		0, (struct sockaddr *)&dest_addr,sizeof(dest_addr));
	pthread_mutex_unlock(&send_mutex);
	return 0;
}