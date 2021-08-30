#include "tof_data_process.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "tof_acquisition.h"

bool is_file_exists(const std::string& name) 
{
    std::ifstream f(name.c_str());
    return f.good();
}

int filter_point_cloud( const cv::Mat &depth_map, TOFAcquisition::PointCloud &src_cloud)
{
	const uchar* data = depth_map.ptr<uchar>(0);
	for (auto it = src_cloud.begin(); it != src_cloud.end();)
    {
		int index = it->index;
		if(*(data + index) == 0)
		{
			it = src_cloud.erase(it);
		}
        else
		{
            ++it;
        }
    }
}

int compute_point_count(const TOFAcquisition::PointCloud &bg_cloud, TOFAcquisition::PointCloud &src_cloud)
{
	int result = 0;
	float min_dist = 1000;
	float temp_dist = 0;
	for (size_t i = 0; i < src_cloud.size(); i++)
	{
		min_dist = 1000;
		for (size_t j = 0; j < bg_cloud.size(); j++)
		{
			temp_dist = abs(src_cloud[i].z - bg_cloud[j].z);
			if (temp_dist < min_dist)
			{
				min_dist = temp_dist;
			}
		}
		if(min_dist >= 0.1f)
		{
			result++;
		}
	}
	return result;
}

void init_background(const std::string &bg_path, cv::Mat &bg_map)
{
	if(!is_file_exists(bg_path))
	{
		cv::Mat filter_map;
		cv::medianBlur(bg_map, filter_map, 5);
		cv::imwrite(bg_path, filter_map);
		bg_map = filter_map;
	}
	else
	{
		bg_map = cv::imread(bg_path.c_str());
	}
}

int compute_depth_map(const cv::Mat &bg_map, const cv::Mat &depth_map)
{
	int result = 0;
	int diff = 0;
	for (int i = 0; i < depth_map.rows; i++)
    {
        const uchar* bg_data = bg_map.ptr<uchar>(i);
        const uchar* data = depth_map.ptr<uchar>(i);
        for (int j=0; j<depth_map.cols; j++)
        {
			diff = abs(bg_data[j] - data[j]);
			if(diff > 50)
			{
				result++;
			}
        }
    }
	return result;
}

int vote_in_out(const std::vector<int> &point_cout_list)
{
	int result = -1;
	int in_count = 0;
	int out_count = 0;
	int diff_count = 0;
	for(size_t i = 1; i < point_cout_list.size(); i++)
	{
		diff_count = point_cout_list[i] - point_cout_list[i-1];
		if(diff_count > 30)
		{
			in_count++;
		}
		else if(diff_count < -30)
		{
			out_count++;
		}
	}

	if(in_count > out_count)
	{
		result = 0;
	}
	else if(in_count < out_count)
	{
		result = 1;
	}
	return result;
}

int get_in_out(const std::vector<int> &result_list)
{
	int in_count = 0;
	int out_count = 0;
	for(size_t i = 1; i < result_list.size(); i++)
	{
		if(result_list[i] == 0)
		{
			in_count++;
		}
		else if(result_list[i] == 1)
		{
			out_count++;
		}
	}
	if(in_count > out_count)
	{
		return 0;
	}
	else if(in_count < out_count)
	{
		return 1;
	}
	else
	{
		return -1;
	}
}