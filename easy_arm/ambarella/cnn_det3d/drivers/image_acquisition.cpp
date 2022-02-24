#include "image_acquisition.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <sched.h>

#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <assert.h>

#include <basetypes.h>
#include <iav_ioctl.h>
#include "datatx_lib.h"


#ifndef VERIFY_BUFFERID
#define VERIFY_BUFFERID(x)	do {		\
			if ((x) < 0 || ((x) >= IAV_MAX_CANVAS_BUF_NUM)) {	\
				printf("Invalid canvas buffer id %d.\n", (x));	\
				return -1;	\
			}	\
		} while (0)
#endif

typedef enum {
	CAPTURE_NONE = 255,
	CAPTURE_PREVIEW_BUFFER = 0,
	CAPTURE_ME1_BUFFER,
	CAPTURE_ME0_BUFFER,
	CAPTURE_RAW_BUFFER,
	CAPTURE_PYRAMID_BUFFER,
	CAPTURE_TYPE_NUM,
} CAPTURE_TYPE;

#define MB_UNIT (16)
#define MAX_YUV_BUFFER_SIZE		(4096*3000)		// 4096x3000
#define MAX_ME_BUFFER_SIZE		(MAX_YUV_BUFFER_SIZE / 16)	// 1/16 of 4096x3000
#define MAX_FRAME_NUM				(120)
#define MARGIN	(30)
#define HWTIMER_OUTPUT_FREQ	(90000)
#define NAME_SIZE	320

#define YUVCAP_PORT					(2024)

typedef enum {
	AUTO_FORMAT = -1,
	YUV420_IYUV = 0,	// Pattern: YYYYYYYYUUVV
	YUV420_YV12 = 1,	// Pattern: YYYYYYYYVVUU
	YUV420_NV12 = 2,	// Pattern: YYYYYYYYUVUV
	YUV422_YU16 = 3,	// Pattern: YYYYYYYYUUUUVVVV
	YUV422_YV16 = 4,	// Pattern: YYYYYYYYVVVVUUUU
	YUV422_NV16 = 5,	// Pattern: YYYYYYYYUVUVUVUV
	YUV444 = 6,
	YUV_FORMAT_TOTAL_NUM,
	YUV_FORMAT_FIRST = YUV420_IYUV,
	YUV_FORMAT_LAST = YUV_FORMAT_TOTAL_NUM,
} YUV_FORMAT;

typedef struct {
	u8 *in;
	u8 *u;
	u8 *v;
	u64 row;
	u64 col;
	u64 pitch;
} yuv_neon_arg;

int fd_iav;

static int transfer_method = TRANS_METHOD_NFS;
static int port = YUVCAP_PORT;

static int vinc_id = 0;
static u32 current_channel = 0;
static int current_buffer = -1;
static int capture_select = 0;
static int non_block_read = 0;

static int yuv_buffer_id = 0;
static int yuv_format = 0;

static int pyramid_buffer_map = 0;

static int dump_canvas_map = 0;
static int me1_buffer_id = 0;
static int me0_buffer_id = 0;
static int frame_count = 1;
static int quit_capture = 0;
static int verbose = 0;
static int info_only = 0;
static int delay_frame_cap_data = 0;
static int G_multi_vin_num = 1;
static int G_canvas_num = 1;
static int dump_canvas_flag = 0;

const char *default_filename_nfs = "/mnt/media/test.yuv";
const char *default_filename_tcp = "media/test";
const char *default_host_ip_addr = "10.0.0.1";
const char *default_filename;
static char filename[256];
static int fd_yuv[IAV_MAX_CANVAS_BUF_NUM];
static int fd_me[IAV_MAX_CANVAS_BUF_NUM];
static int fd_raw = -1;
static int fd_pyramid[IAV_MAX_PYRAMID_LAYERS];

static u8 *dsp_mem = NULL;
static u32 dsp_size = 0;

static u8* dsp_canvas_yuv_buf_mem[IAV_MAX_CANVAS_BUF_NUM];
static u32 dsp_canvas_yuv_buf_size[IAV_MAX_CANVAS_BUF_NUM];
static u32 dsp_canvas_yuv_offset[IAV_MAX_CANVAS_BUF_NUM];
static u8* dsp_canvas_me_buf_mem[IAV_MAX_CANVAS_BUF_NUM];
static u32 dsp_canvas_me_buf_size[IAV_MAX_CANVAS_BUF_NUM];
static u32 dsp_canvas_me_offset[IAV_MAX_CANVAS_BUF_NUM];
static u8* dsp_pyramid_buf = NULL;
static u32 dsp_pyramid_buf_size = 0;
static u8* gdma_dst_buf = NULL;
static u32 gdma_dst_buf_size = 0;
static u32 gdma_dst_buf_pid = 0;
static u32 dsp_buf_mapped = 0;
static u8* pyramid_pool_buf = NULL;
static u32 pyramid_pool_buf_size = 0;
static u32 pyramid_pool_buf_mapped = 0;
static u32 pyramid_manual_feed = 0;
static u32 pts_intval[IAV_MAX_CANVAS_BUF_NUM];

static int decode_mode = 0;

static struct timeval pre = {0, 0}, curr = {0, 0};

static struct ImageBuffer image_buffer;
volatile int run_camera = 0; 

extern "C"{
extern void chrome_convert(yuv_neon_arg *);
extern void chrome_UV22_convert_to_UV44(yuv_neon_arg *);
extern void chrome_UV20_convert_to_UV44(yuv_neon_arg *);
}

//first second value must in format "x~y" if delimiter is '~'
static int get_two_unsigned_int(char *name, u32 *first, u32 *second,
	char delimiter)
{
	char tmp_string[16];
	char * separator;

	separator = strchr(name, delimiter);
	if (!separator) {
		printf("range should be like a%cb \n", delimiter);
		return -1;
	}

	strncpy(tmp_string, name, separator - name);
	tmp_string[separator - name] = '\0';
	*first = atoi(tmp_string);
	strncpy(tmp_string, separator + 1,  name + strlen(name) -separator);
	*second = atoi(tmp_string);

	return 0;
}

static int map_canvas_buffers(void)
{
	struct iav_querymem query_mem;
	struct iav_mem_canvas_info *canvas_info;
	int i;

	query_mem.mid = IAV_MEM_CANVAS;
	query_mem.arg.canvas.id_map = dump_canvas_map;
	if (ioctl(fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
		perror("IAV_IOC_QUERY_MEMBLOCK");
		return -1;
	}
	canvas_info = &query_mem.arg.canvas;

	memset(dsp_canvas_yuv_buf_mem, 0, sizeof(dsp_canvas_yuv_buf_mem));
	memset(dsp_canvas_me_buf_mem, 0, sizeof(dsp_canvas_me_buf_mem));
	for (i = 0; i < IAV_MAX_CANVAS_BUF_NUM; ++i) {
		if (dump_canvas_map & (1 << i)) {
			dsp_canvas_yuv_buf_size[i] = canvas_info->yuv[i].length;
			dsp_canvas_yuv_offset[i] = canvas_info->yuv[i].offset;
			if (canvas_info->yuv[i].addr) {
				dsp_canvas_yuv_buf_mem[i] = (u8*)mmap(NULL, dsp_canvas_yuv_buf_size[i],
					PROT_READ, MAP_SHARED, fd_iav, canvas_info->yuv[i].addr);
				if (dsp_canvas_yuv_buf_mem[i] == MAP_FAILED) {
					perror("mmap failed\n");
					return -1;
				}
			}
			dsp_canvas_me_buf_size[i] = canvas_info->me[i].length;
			dsp_canvas_me_offset[i] = canvas_info->me[i].offset;
			if (canvas_info->me[i].addr) {
				dsp_canvas_me_buf_mem[i] = (u8*)mmap(NULL, dsp_canvas_me_buf_size[i],
					PROT_READ, MAP_SHARED, fd_iav, canvas_info->me[i].addr);
				if (dsp_canvas_me_buf_mem[i] == MAP_FAILED) {
					perror("mmap failed\n");
					return -1;
				}
			}
		}
	}

	return 0;
}

static int map_dsp_buffer(void)
{
	struct iav_querymem query_mem;
	struct iav_mem_part_info *part_info;

	memset(&query_mem, 0, sizeof(query_mem));
	query_mem.mid = IAV_MEM_PARTITION;
	part_info = &query_mem.arg.partition;
	part_info->pid = IAV_PART_DSP;
	if (ioctl(fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
		perror("IAV_IOC_QUERY_MEMBLOCK");
		return -1;
	}

	dsp_size = part_info->mem.length;
	dsp_mem = (u8*)mmap(NULL, dsp_size, PROT_READ, MAP_SHARED, fd_iav,
		part_info->mem.addr);
	if (dsp_mem == MAP_FAILED) {
		perror("mmap IAV_PART_DSP failed\n");
		return -1;
	}
	dsp_buf_mapped = 1;

	memset(&query_mem, 0, sizeof(query_mem));
	query_mem.mid = IAV_MEM_PARTITION;
	part_info = &query_mem.arg.partition;
	part_info->pid = IAV_PART_PYRAMID_POOL;
	if (ioctl(fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
		perror("IAV_IOC_QUERY_MEMBLOCK");
		return -1;
	}
	dsp_pyramid_buf_size = part_info->mem.length;
	if (dsp_pyramid_buf_size) {
		printf("user buffer size(0x%x) > 0, GDMA will be used\n", dsp_pyramid_buf_size);
		dsp_pyramid_buf = (u8*)mmap(NULL, dsp_pyramid_buf_size, PROT_READ | PROT_WRITE, MAP_SHARED,
			fd_iav, part_info->mem.addr);
		if (dsp_pyramid_buf == MAP_FAILED) {
			perror("mmap IAV_PART_PYRAMID_POOL failed\n");
			return -1;
		}
	}

	if ((capture_select == CAPTURE_PYRAMID_BUFFER) && pyramid_manual_feed) {
		memset(&query_mem, 0, sizeof(query_mem));
		query_mem.mid = IAV_MEM_PARTITION;
		part_info = &query_mem.arg.partition;
		part_info->pid = IAV_PART_PYRAMID_POOL;
		if (ioctl(fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
			perror("IAV_IOC_QUERY_MEMBLOCK");
			return -1;
		}

		pyramid_pool_buf_size = part_info->mem.length;
		if (pyramid_pool_buf_size) {
			pyramid_pool_buf = (u8*)mmap(NULL, pyramid_pool_buf_size, PROT_READ, MAP_SHARED, fd_iav,
				part_info->mem.addr);
			if (pyramid_pool_buf == MAP_FAILED) {
				perror("mmap IAV_PART_PYRAMID_POOL failed\n");
				return -1;
			}
			pyramid_pool_buf_mapped = 1;
		}
	}

	return 0;
}

static int alloc_gdma_dst_buf(u32 size)
{
	struct iav_alloc_mem_part alloc_mem_part;
	struct iav_querymem query_mem;
	struct iav_mem_part_info *part_info;

	alloc_mem_part.length = size;
	alloc_mem_part.enable_cache = 1;
	if (ioctl(fd_iav, IAV_IOC_ALLOC_ANON_MEM_PART, &alloc_mem_part) < 0) {
		perror("IAV_IOC_ALLOC_ANON_MEM_PART");
		return -1;
	}
	gdma_dst_buf_pid = alloc_mem_part.pid;

	memset(&query_mem, 0, sizeof(query_mem));
	query_mem.mid = IAV_MEM_PARTITION;
	part_info = &query_mem.arg.partition;
	part_info->pid = gdma_dst_buf_pid;
	if (ioctl(fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
		perror("IAV_IOC_QUERY_MEMBLOCK");
		return -1;
	}

	gdma_dst_buf_size = part_info->mem.length;
	if (gdma_dst_buf_size) {
		gdma_dst_buf = (u8*)mmap(NULL, gdma_dst_buf_size, PROT_READ | PROT_WRITE, MAP_SHARED,
			fd_iav, part_info->mem.addr);
		if (gdma_dst_buf == MAP_FAILED) {
			perror("mmap gdma dst buffer failed\n");
			return -1;
		}
	}

	return 0;

}

static int free_gdma_dst_buf(void)
{
	struct iav_free_mem_part free_mem_part;

	if (gdma_dst_buf && gdma_dst_buf_size) {
		munmap(gdma_dst_buf, gdma_dst_buf_size);
	}
	gdma_dst_buf = NULL;
	gdma_dst_buf_size = 0;

	free_mem_part.pid = gdma_dst_buf_pid;
	if (ioctl(fd_iav, IAV_IOC_FREE_MEM_PART, &free_mem_part) < 0) {
		perror("IAV_IOC_FREE_MEM_PART");
		return -1;
	}
	gdma_dst_buf_pid = 0;

	return 0;
}

static int map_buffer(void)
{
	int ret = 0;

	if (dump_canvas_flag) {
		ret = map_canvas_buffers();
	} else {
		ret = map_dsp_buffer();
	}

	return ret;
}

static u8* get_buffer_base(int buf_id, int me_flag)
{
	u8* addr = NULL;

	if (buf_id < 0 || buf_id >= IAV_MAX_CANVAS_BUF_NUM) {
		printf("Invaild canvas buf ID %d!\n", buf_id);
		return NULL;
	}
	if (dump_canvas_map & (1 << buf_id)) {
		if (me_flag) {
			addr = dsp_canvas_me_buf_mem[buf_id];
		} else {
			addr = dsp_canvas_yuv_buf_mem[buf_id];
		}
	} else {
		if ((capture_select == CAPTURE_PYRAMID_BUFFER) && pyramid_manual_feed) {
			addr = pyramid_pool_buf;
		} else {
			addr = dsp_mem;
		}
	}
	return addr;
}

static int save_yuv_luma_buffer(int buf_id, u8* input, u8* output,
	struct iav_yuv_cap *yuv_cap)
{
	int i;
	u8 *in = NULL;
	u8 *out = NULL;

	if (yuv_cap->pitch < yuv_cap->width) {
		printf("pitch size smaller than width!\n");
		return -1;
	}

	if (input == NULL || output == NULL) {
		printf("Invalid pointer NULL!");
		return -1;
	}

	if (yuv_cap->pitch == yuv_cap->width) {
		memcpy(output, input, yuv_cap->width * yuv_cap->height);
	} else {
		in = input;
		out = output;
		for (i = 0; i < yuv_cap->height; i++) {		//row
			memcpy(out, in, yuv_cap->width);
			in += yuv_cap->pitch;
			out += yuv_cap->width;
		}
	}

	return 0;
}

static int get_yuv_format(int format, struct iav_yuv_cap *yuv_cap)
{
	int data_format = format;

	if (data_format == AUTO_FORMAT) {
		if (yuv_cap->format == IAV_YUV_FORMAT_YUV420) {
			data_format = YUV420_NV12;
		} else if (yuv_cap->format == IAV_YUV_FORMAT_YUV422) {
			data_format = YUV422_NV16;
		} else {
			printf("Unknown YUV format: %d\n", yuv_cap->format);
		}
	}

	return data_format;
}

static int save_yuv_chroma_buffer(int buf_id, u8* input, u8* output,
	struct iav_yuv_cap *yuv_cap, u8 gdma_flag)
{
	int width, height, pitch;
	u8 *uv_addr = NULL;
	int i, format;
	yuv_neon_arg yuv;
	int ret = 0;

	if (input == NULL || output == NULL) {
		printf("Invalid pointer NULL!");
		return -1;
	}

	uv_addr = input;
	format = get_yuv_format(yuv_format, yuv_cap);
	width = yuv_cap->width;
	height = yuv_cap->height;
	if (gdma_flag == 1) {
		pitch = yuv_cap->width;
	} else {
		pitch = yuv_cap->pitch;
	}

	// input yuv is uv interleaved with padding (uvuvuvuv.....)
	if (yuv_cap->format == IAV_YUV_FORMAT_YUV420) {
		yuv.in = uv_addr;
		yuv.row = height / 2 ;
		yuv.col = width;
		yuv.pitch = pitch;
		if (format == YUV420_YV12) {
			// YV12 format (YYYYYYYYVVUU)
			yuv.u = output + width * height / 4;
			yuv.v = output;
			chrome_convert(&yuv);
		} else if (format == YUV420_IYUV) {
			// IYUV (I420) format (YYYYYYYYUUVV)
			yuv.u = output;
			yuv.v = output + width * height / 4;
			chrome_convert(&yuv);
		} else if (format == YUV444) {
			yuv.u = output;
			yuv.v = output + width * height;
			chrome_UV20_convert_to_UV44(&yuv);
		} else {
			if (format != YUV420_NV12) {
				printf("Change output format back to NV12 for encode!\n");
				format = YUV420_NV12;
			}
			if (gdma_flag == 1) {
				printf("Unexpected! No need chroma convert!\n");
				ret = -1;
			}
			// NV12 format (YYYYYYYYUVUV)
			input = uv_addr;
			for (i = 0; i < height / 2; ++i) {
				memcpy(output + i * width, input + i * pitch,
					width);
			}
		}
	} else if (yuv_cap->format == IAV_YUV_FORMAT_YUV422) {
		yuv.in = uv_addr;
		yuv.row = height;
		yuv.col = width;
		yuv.pitch = pitch;
		if (format == YUV422_YU16) {
			yuv.u = output;
			yuv.v = output + width * height / 2;
			chrome_convert(&yuv);
		} else if (format == YUV422_NV16){
			// NV16 format (YYYYYYYYUVUVUVUV)
			if (gdma_flag == 1) {
				printf("Unexpected! No need chroma convert!\n");
				ret = -1;
			}
			input = uv_addr;
			for (i = 0; i < height; ++i) {
				memcpy(output + i * width, input + i * pitch, width);
			}
		} else if (format == YUV444) {
			yuv.u = output;
			yuv.v = output + width * height;
			chrome_UV22_convert_to_UV44(&yuv);
		} else {
			if (format != YUV422_YV16) {
				printf("Change output format back to YV16 for preview!\n");
				format = YUV422_YV16;
			}
			yuv.u = output + width * height / 2;
			yuv.v = output;
			chrome_convert(&yuv);
		}
	} else {
		printf("Error: Unsupported YUV input format!\n");
		ret = -1;
	}

	return ret;
}

static int save_yuv_data(int fd, int buf_id, struct iav_yuv_cap *yuv_cap,
	u8 *luma, u8 *chroma)
{
	static int pts_prev = 0, pts = 0;
	int luma_size, chroma_size;
	u8 *base = NULL;
	u8 * luma_addr = NULL, *chroma_addr = NULL;
	int format = get_yuv_format(yuv_format, yuv_cap);

	luma_size = yuv_cap->width * yuv_cap->height;
	if (yuv_cap->format == IAV_YUV_FORMAT_YUV420) {
		if (format == YUV444) {
			chroma_size = luma_size << 1;
		} else {
			chroma_size = luma_size >> 1;
		}
	} else if (yuv_cap->format == IAV_YUV_FORMAT_YUV422) {
		if (format == YUV444) {
			chroma_size = luma_size << 1;
		} else {
			chroma_size = luma_size;
		}
	} else {
		printf("Error: Unrecognized yuv data format from DSP!\n");
		return -1;
	}

	/* Save YUV data into memory */
	if (verbose) {
		gettimeofday(&curr, NULL);
		pre = curr;
	}

	if (gdma_dst_buf_size && dsp_buf_mapped) {
		base = gdma_dst_buf;
		luma_addr = base;
		chroma_addr = luma_addr + yuv_cap->width * yuv_cap->height;

		/* Save UV data into another memory if it needs convert. */
		if (format == YUV420_YV12 || format == YUV420_IYUV ||
			format == YUV444 || format == YUV422_YU16 ||
			format == YUV422_YV16) {
			if (save_yuv_chroma_buffer(buf_id, chroma_addr, chroma, yuv_cap, 1) < 0) {
				perror("Failed to save chroma data into buffer !\n");
				return -1;
			}
			chroma_addr = chroma;
		}
	} else {
		base = get_buffer_base(buf_id, 0);
		luma_addr = base + yuv_cap->y_addr_offset;
		chroma_addr = base + yuv_cap->uv_addr_offset;

		if (save_yuv_luma_buffer(buf_id, luma_addr, luma, yuv_cap) < 0) {
			perror("Failed to save luma data into buffer !\n");
			return -1;
		}
		luma_addr = luma;
		if (verbose) {
			gettimeofday(&curr, NULL);
			printf("22. Save Y [%06ld us].\n", (curr.tv_sec - pre.tv_sec) *
				1000000 + (curr.tv_usec - pre.tv_usec));
			pre = curr;
		}
		if (save_yuv_chroma_buffer(buf_id, chroma_addr, chroma, yuv_cap, 0) < 0) {
			perror("Failed to save chroma data into buffer !\n");
			return -1;
		}
		chroma_addr = chroma;
	}

	if (verbose) {
		gettimeofday(&curr, NULL);
		printf("33. Save UV [%06ld us].\n", (curr.tv_sec - pre.tv_sec) *
			1000000 + (curr.tv_usec - pre.tv_usec));
	}

	/* Write YUV data from memory to file */
	if (amba_transfer_write(fd, luma_addr, luma_size, transfer_method) < 0) {
		perror("Failed to save luma data into file !\n");
		return -1;
	}
	if (amba_transfer_write(fd, chroma_addr, chroma_size, transfer_method) < 0) {
		perror("Failed to save chroma data into file !\n");
		return -1;
	}

	if (verbose) {
		pts = yuv_cap->mono_pts;
		printf("BUF [%d] Y 0x%08x, UV 0x%08x, pitch %u, %ux%u = Seqnum[%u], "
			"PTS [%u-%u].\n", buf_id, (u32)yuv_cap->y_addr_offset,
			(u32)yuv_cap->uv_addr_offset, yuv_cap->pitch, yuv_cap->width,
			yuv_cap->height, yuv_cap->seq_num, pts, (pts - pts_prev));
		pts_prev = pts;
	}

	return 0;
}

static int dump_yuv()
{
	/* TO DO */
	printf("Canvas mem dump hasn't been supported as source buffer auto stop hasn't been implemented yet.\n");
	return -1;
#if 0
	int i, buf;
	char yuv_file[320];
	u8 *yuv = NULL;
	u8 *p_in = NULL;
	u8 *p_out = NULL;
	yuv_neon_arg yuv_neon;

	int width = 0;
	int height = 0;
	int max_height = 0;
	int pitch = 0;
	int one_line = 0;
	int r_size = 0;
	int w_size = 0;
	int remain_size = 0;

	enum iav_yuv_format canvas_format;
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *yuv_cap;

	for (buf = 0; buf < IAV_MAX_CANVAS_BUF_NUM; ++buf) {
		if (dump_canvas_map & (1 << buf)) {
			/* It's also supported to use 'IAV_DESC_YUV' here. */
			query_desc.qid = IAV_DESC_CANVAS;
			query_desc.arg.canvas.canvas_id = buf;
			query_desc.arg.canvas.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
			if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
				if (errno == EINTR) {
					continue;		/* back to for() */
				} else {
					perror("IAV_IOC_QUERY_DESC");
					goto dump_yuv_error_exit;
				}
			}

			yuv_cap = &query_desc.arg.canvas.yuv;
			width = yuv_cap->width;
			height = yuv_cap->height;
			/* FIXME: use max height the same as height */
			max_height = height;
			pitch = yuv_cap->pitch;
			canvas_format = yuv_cap->format;

			if (fd_yuv[buf] < 0) {
				memset(yuv_file, 0, sizeof(yuv_file));
				sprintf(yuv_file, "%s_canvas%d_%dx%d.yuv", filename, buf,
					width, height);
				fd_yuv[buf] = amba_transfer_open(yuv_file, transfer_method,
					port++);
				if (fd_yuv[buf] < 0) {
					printf("Cannot open file [%s].\n", yuv_file);
					continue;
				}
			}

			if (canvas_format ==  IAV_YUV_FORMAT_YUV420) {
				w_size = (width * height) * 3 / 2;
				r_size = (pitch * max_height) * 3 / 2;
			} else if (canvas_format ==  IAV_YUV_FORMAT_YUV422) {
				w_size = (width * height) * 2;
				r_size = (pitch * max_height) * 2;
			} else {
				printf("Invalid canvas format [%u]!\n", canvas_format);
				return -1;
			}

			p_in = dsp_canvas_yuv_buf_mem[buf] + dsp_canvas_yuv_offset[buf];
			remain_size = dsp_canvas_yuv_buf_size[buf] - dsp_canvas_yuv_offset[buf];

			if (yuv == NULL) {
				yuv = malloc(w_size);
				if (yuv == NULL) {
					printf("Not enough memory for preview dump !\n");
					goto dump_yuv_error_exit;
				}
			}

			for (i = 0; remain_size >= r_size; i++) { // memcpy pre frame in NV12 format
				p_out = yuv;
				/* copy Y */
				for (one_line = 0; one_line < height; one_line++) {
					memcpy(p_out, p_in, width);
					p_in += pitch;
					p_out += width;
				}
				/* consider max_height here */
				p_in += (max_height - height) * pitch;
				if (canvas_format == IAV_YUV_FORMAT_YUV420) {
					for (one_line = 0; one_line < height / 2; one_line++) {
						memcpy(p_out, p_in, width);
						p_in += pitch;
						p_out += width;
					}
					p_in += ((max_height - height) * pitch) / 2;
				} else if (canvas_format == IAV_YUV_FORMAT_YUV422) {
					yuv_neon.in = p_in;
					yuv_neon.row = height;
					yuv_neon.col = width;
					yuv_neon.pitch = pitch;
					yuv_neon.u = p_out;
					yuv_neon.v = p_out + width * height / 2;
					chrome_convert(&yuv_neon);

					p_in += max_height * pitch;
				}
				remain_size -= r_size;
				if (amba_transfer_write(fd_yuv[buf], yuv, w_size, transfer_method) < 0) {
					perror("Failed to save yuv all memory data into file !\n");
					goto dump_yuv_error_exit;
				}
			}
			printf("Dump YUV: resolution %dx%d in %s format, total frame num "
				"[%d], ""file: %s\n", width, height,
				(canvas_format == IAV_YUV_FORMAT_YUV420) ? "NV12" : "YU16",
				i, yuv_file);
		}
	}

	if (yuv) {
		free(yuv);
	}
	return 0;

dump_yuv_error_exit:
	if (yuv) {
		free(yuv);
	}
	return -1;
#endif
}

static int query_yuv(int buf_id, int count)
{
	int i = 0, buf;
	int non_stop = 0, curr_format;
	char format[32];
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *yuv_cap;
	int rval = 0;

	u64 pts_prev[IAV_MAX_CANVAS_BUF_NUM] = {0};
	u64 pts[IAV_MAX_CANVAS_BUF_NUM] = {0};
	u16 intval = 0;

	if (count == 0) {
		non_stop = 1;
	}

	while ((i < count || non_stop) && !quit_capture) {
		for (buf = 0; buf < G_canvas_num; ++buf) {
			if (buf_id & (1 << buf)) {
				memset(&query_desc, 0, sizeof(query_desc));
				query_desc.qid = IAV_DESC_CANVAS;
				query_desc.arg.canvas.canvas_id = buf;
				if (!non_block_read) {
					query_desc.arg.canvas.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
				} else {
					query_desc.arg.canvas.non_block_flag |= IAV_BUFCAP_NONBLOCK;
				}

				if (verbose) {
					gettimeofday(&curr, NULL);
					pre = curr;
				}
				if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
					if (errno == EINTR) {
						continue;		/* back to for() */
					} else {
						perror("IAV_IOC_QUERY_DESC");
						rval = -1;
						break;
					}
				}
				if (verbose) {
					gettimeofday(&curr, NULL);
					printf("Query CANVAS DESC [%06ld us].\n", 1000000 *
						(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));
				}

				yuv_cap = &query_desc.arg.canvas.yuv;
				if ((yuv_cap->y_addr_offset == 0) || (yuv_cap->uv_addr_offset == 0)) {
					printf("YUV buffer [%d] address is NULL! Skip to next!\n", buf);
					continue;
				}

				curr_format = get_yuv_format(yuv_format, yuv_cap);
				if (yuv_cap->format == IAV_YUV_FORMAT_YUV422) {
					if (curr_format == YUV422_YU16) {
						sprintf(format, "YU16");
					} else if (curr_format == YUV422_YV16) {
						sprintf(format, "YV16");
					} else if (curr_format == YUV422_NV16) {
						sprintf(format, "NV16");
					} else {
						sprintf(format, "YV16");
					}
				} else if (yuv_cap->format == IAV_YUV_FORMAT_YUV420) {
					switch (curr_format) {
					case YUV420_YV12:
						sprintf(format, "YV12");
						break;
					case YUV420_NV12:
						sprintf(format, "NV12");
						break;
					case YUV420_IYUV:
						sprintf(format, "IYUV");
						break;
					default:
						sprintf(format, "NV12");
						break;
					}
				} else {
					sprintf(format, "Unknown [%d]", yuv_cap->format);
				}
				if (verbose) {
					printf("Capture YUV cavas(buffer)%d: size[%dx%d] in %s format Seqnum[%u] PTS [%llu-%llu]. \n",
							buf, yuv_cap->width, yuv_cap->height, format, yuv_cap->seq_num, pts[buf], (pts[buf] - pts_prev[buf]));
				}

				pts[buf] = yuv_cap->mono_pts;
				if (pts_prev[buf]) {
					intval = pts[buf] - pts_prev[buf];
					if ((intval > pts_intval[buf] + MARGIN) ||
						(intval < pts_intval[buf] - MARGIN)) {
						printf("Error! Discontinuous mono PTS, frame lost for canvas[%d]!\n", buf);
					}
				}
				pts_prev[buf] = pts[buf];
			}
		}
		if (rval < 0) {
			break;
		}
		++i;
	}

	return rval;
}

static int query_pyramid(int pyramid_map, int count)
{
	int i = 0, buf;
	int non_stop = 0, curr_format;
	char format[32];
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *pyramid_cap;
	struct iav_feed_pyramid feed_pyramid;
	int rval = 0;

	static int pts_prev = 0, pts = 0;

	if (count == 0) {
		non_stop = 1;
	}

	while ((i < count || non_stop) && !quit_capture) {
		memset(&query_desc, 0, sizeof(query_desc));
		query_desc.qid = IAV_DESC_PYRAMID;
		query_desc.arg.pyramid.chan_id = current_channel;
		if (!non_block_read) {
			query_desc.arg.pyramid.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
		} else {
			query_desc.arg.pyramid.non_block_flag |= IAV_BUFCAP_NONBLOCK;
		}

		gettimeofday(&curr, NULL);
		pre = curr;

		if (pyramid_manual_feed) {
			feed_pyramid.chan_id = current_channel;
			if (ioctl(fd_iav, IAV_IOC_FEED_PYRAMID_BUF, &feed_pyramid) < 0) {
				perror("IAV_IOC_FEED_PYRAMID_BUF");
				rval = -1;
				break;
			}
		}

		if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
			if (errno == EINTR) {
				continue;		/* back to for() */
			} else {
				perror("IAV_IOC_QUERY_DESC");
				rval = -1;
				break;
			}
		}

		gettimeofday(&curr, NULL);
		printf("Query PYRAMIDS DESC [%06ld us].\n", 1000000 *
			(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));

		for (buf = 0; buf < IAV_MAX_PYRAMID_LAYERS; ++buf) {
			if ((pyramid_map & (1 << buf)) == 0) {
				continue;
			}

			if ((query_desc.arg.pyramid.layers_map & (1 << buf)) == 0) {
				printf("Pyramid channel %d: layer %d is not switched on\n",
					current_channel, buf);
					continue;
			}

			pyramid_cap = &query_desc.arg.pyramid.layers[buf];

			pts = pyramid_cap->mono_pts;

			curr_format = get_yuv_format(yuv_format, pyramid_cap);
			if (pyramid_cap->format == IAV_YUV_FORMAT_YUV420) {
				switch (curr_format) {
				case YUV420_YV12:
					sprintf(format, "YV12");
					break;
				case YUV420_NV12:
					sprintf(format, "NV12");
					break;
				case YUV420_IYUV:
					sprintf(format, "IYUV");
					break;
				default:
					sprintf(format, "IYUV");
					break;
				}
			} else {
				sprintf(format, "Unknown [%d]", pyramid_cap->format);
			}
			printf("Capture_Pyramid_buffer: Pyramid layer %d for chan %d "
				"resolution %dx%d in %s format Seqnum[%u] PTS [%u-%u].\n", buf, current_channel,
				pyramid_cap->width, pyramid_cap->height, format, pyramid_cap->seq_num, pts, (pts - pts_prev));
			pts_prev = pts;

		}
		if (rval < 0) {
			break;
		}

		++i;
	}

	return rval;
}

static int capture_yuv(int buf_id, int count)
{
	int i, buf, save[IAV_SRCBUF_NUM];
	int write_flag[IAV_SRCBUF_NUM];
	char yuv_file[320];
	int non_stop = 0, curr_format;
	u8 *luma = NULL, *chroma = NULL;
	char format[32];
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *yuv_cap;
	struct iav_yuv_cap yuv_cap_cache;
	struct iav_gdma_copy gdma_copy = {0};
	int yuv_buffer_size = 0;
	int rval = 0;

	do {
		luma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
		if (luma == NULL) {
			printf("Not enough memory for preview capture !\n");
			rval = -1;
			break;
		}
		chroma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
		if (chroma == NULL) {
			printf("Not enough memory for preview capture !\n");
			rval = -1;
			break;
		}
		memset(save, 0, sizeof(save));
		memset(write_flag, 0, sizeof(write_flag));
		memset(luma, 1, MAX_YUV_BUFFER_SIZE);
		memset(chroma, 1, MAX_YUV_BUFFER_SIZE);

		if (count == 0) {
			non_stop = 1;
		}

		i = 0;
		while ((i < count || non_stop) && !quit_capture) {
			for (buf = 0; buf < G_canvas_num; ++buf) {
				if (buf_id & (1 << buf)) {
					memset(&query_desc, 0, sizeof(query_desc));
					query_desc.qid = IAV_DESC_CANVAS;
					query_desc.arg.canvas.canvas_id = buf;
					if (!non_block_read) {
						query_desc.arg.canvas.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
					} else {
						query_desc.arg.canvas.non_block_flag |= IAV_BUFCAP_NONBLOCK;
					}

					if (verbose) {
						gettimeofday(&curr, NULL);
						pre = curr;
					}
					if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
						if (errno == EINTR) {
							continue;		/* back to for() */
						} else {
							perror("IAV_IOC_QUERY_DESC");
							rval = -1;
							break;
						}
					}
					if (verbose) {
						gettimeofday(&curr, NULL);
						printf("11. Query DESC [%06ld us].\n", 1000000 *
							(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));
					}
					yuv_cap = &query_desc.arg.canvas.yuv;
					if ((yuv_cap->y_addr_offset == 0) || (yuv_cap->uv_addr_offset == 0)) {
						printf("YUV buffer [%d] address is NULL! Skip to next!\n", buf);
						continue;
					}

					if (fd_yuv[buf] < 0) {
						memset(yuv_file, 0, sizeof(yuv_file));
						sprintf(yuv_file, "%s_canvas%d_%dx%d.yuv", filename, buf,
							yuv_cap->width, yuv_cap->height);
						if (fd_yuv[buf] < 0) {
							fd_yuv[buf] = amba_transfer_open(yuv_file,
								transfer_method, port++);
						}
						if (fd_yuv[buf] < 0) {
							printf("Cannot open file [%s].\n", yuv_file);
							continue;
						}
					}

					yuv_buffer_size = yuv_cap->pitch * ROUND_UP(yuv_cap->height, 16) * 2;
					if (alloc_gdma_dst_buf(yuv_buffer_size)) {
						rval = -1;
						break;
					}

					if (verbose) {
						gettimeofday(&curr, NULL);
						pre = curr;
					}

					gdma_copy.src_offset = yuv_cap->y_addr_offset;
					gdma_copy.dst_offset = 0;
					gdma_copy.src_pitch = yuv_cap->pitch;
					gdma_copy.dst_pitch = yuv_cap->width;
					gdma_copy.width = yuv_cap->width;
					gdma_copy.height = yuv_cap->height;
					gdma_copy.src_mmap_type = IAV_PART_DSP;
					gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
					if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
						perror("IAV_IOC_GDMA_COPY");
					}

					gdma_copy.src_offset = yuv_cap->uv_addr_offset;
					gdma_copy.dst_offset = yuv_cap->width * yuv_cap->height;
					gdma_copy.src_pitch = yuv_cap->pitch;
					gdma_copy.dst_pitch = yuv_cap->width;
					gdma_copy.width = yuv_cap->width;
					gdma_copy.height = yuv_cap->height;
					gdma_copy.src_mmap_type = IAV_PART_DSP;
					gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
					if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
						perror("IAV_IOC_GDMA_COPY");
					}

					if (verbose) {
						gettimeofday(&curr, NULL);
						printf("12. GDMA copy [%06ld us].\n", 1000000 *
							(curr.tv_sec - pre.tv_sec) + (curr.tv_usec - pre.tv_usec));
					}

					if (delay_frame_cap_data) {
						if (write_flag[buf] == 0) {
							write_flag[buf] = 1;
							yuv_cap_cache = *yuv_cap;
						} else {
							write_flag[buf] = 0;
							if (save_yuv_data(fd_yuv[buf], buf, &yuv_cap_cache, luma, chroma) < 0) {
								printf("Failed to save YUV data of buf [%d].\n", buf);
								rval = -1;
								break;
							}
						}
					} else {
						if (save_yuv_data(fd_yuv[buf], buf, yuv_cap, luma, chroma) < 0) {
							printf("Failed to save YUV data of buf [%d].\n", buf);
							rval = -1;
							break;
						}
					}

					curr_format = get_yuv_format(yuv_format, yuv_cap);
					if (save[buf] == 0) {
						save[buf] = 1;
						if (yuv_cap->format == IAV_YUV_FORMAT_YUV422) {
							switch (curr_format) {
							case YUV422_YU16:
								sprintf(format, "YU16");
								break;
							case YUV422_YV16:
								sprintf(format, "YV16");
								break;
							case YUV422_NV16:
								sprintf(format, "NV16");
								break;
							case YUV444:
								sprintf(format, "YUV444");
								break;
							default:
								sprintf(format, "YV16");
								break;
							}
						} else if (yuv_cap->format == IAV_YUV_FORMAT_YUV420) {
							switch (curr_format) {
							case YUV420_YV12:
								sprintf(format, "YV12");
								break;
							case YUV420_NV12:
								sprintf(format, "NV12");
								break;
							case YUV420_IYUV:
								sprintf(format, "IYUV");
								break;
							case YUV444:
								sprintf(format, "YUV444");
								break;
							default:
								sprintf(format, "NV12");
								break;
							}
						} else {
							sprintf(format, "Unknown [%d]", yuv_cap->format);
						}
						printf("Delay %d frame capture YUV data.\n", delay_frame_cap_data);
						printf("Capture YUV cavas(buffer)%d: size[%dx%d] in %s format\n",
							buf, yuv_cap->width, yuv_cap->height, format);
					}

					if (gdma_dst_buf_pid) {
						free_gdma_dst_buf();
					}
				}
			}
			if (rval < 0) {
				break;
			}
			++i;
		}
	} while (0);

	if (gdma_dst_buf_pid) {
		free_gdma_dst_buf();
	}
	if (luma) {
		free(luma);
	}
	if (chroma) {
		free(chroma);
	}
	return rval;
}

static int capture_pyramid(int pyramid_map, int count)
{
	int i, buf;
	char pyramid_file[320];
	int non_stop = 0, curr_format;
	u8 * luma = NULL;
	u8 * chroma = NULL;
	char format[32];
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *pyramid_cap;
	struct iav_gdma_copy gdma_copy = {0};
	struct iav_feed_pyramid feed_pyramid;
	int yuv_buffer_size = 0;
	int rval = 0;

	do {
		luma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
		if (luma == NULL) {
			printf("Not enough memory for pyramid capture !\n");
			rval = -1;
			break;
		}
		chroma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
		if (chroma == NULL) {
			printf("Not enough memory for pyramid capture !\n");
			rval = -1;
			break;
		}
		memset(luma, 1, MAX_YUV_BUFFER_SIZE);
		memset(chroma, 1, MAX_YUV_BUFFER_SIZE);

		if (count == 0) {
			non_stop = 1;
		}

		i = 0;
		while ((i < count || non_stop) && !quit_capture) {
			memset(&query_desc, 0, sizeof(query_desc));
			query_desc.qid = IAV_DESC_PYRAMID;
			query_desc.arg.pyramid.chan_id = current_channel;
			if (!non_block_read) {
				query_desc.arg.pyramid.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
			} else {
				query_desc.arg.pyramid.non_block_flag |= IAV_BUFCAP_NONBLOCK;
			}

			if (verbose) {
				gettimeofday(&curr, NULL);
				pre = curr;
			}

			if (pyramid_manual_feed) {
				feed_pyramid.chan_id = current_channel;
				if (ioctl(fd_iav, IAV_IOC_FEED_PYRAMID_BUF, &feed_pyramid) < 0) {
					perror("IAV_IOC_FEED_PYRAMID_BUF");
					rval = -1;
					break;
				}
			}

			if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
				if ((errno == EINTR) || (errno == EAGAIN)) {
					continue;		/* back to for() */
				} else {
					perror("IAV_IOC_QUERY_DESC");
					rval = -1;
					break;
				}
			}
			if (verbose) {
				gettimeofday(&curr, NULL);
				printf("11. Query DESC [%06ld us].\n", 1000000 *
					(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));
			}

			for (buf = 0; buf < IAV_MAX_PYRAMID_LAYERS; ++buf) {
				if ((pyramid_map & (1 << buf)) == 0) {
					continue;
				}

				if ((query_desc.arg.pyramid.layers_map & (1 << buf)) == 0) {
					printf("Pyramid channel %d: layer %d is not switched on\n",
						current_channel, buf);
					continue;
				}

				pyramid_cap = &query_desc.arg.pyramid.layers[buf];
				if (fd_pyramid[buf] < 0) {
					memset(pyramid_file, 0, sizeof(pyramid_file));
					sprintf(pyramid_file, "%s_chan_%d_%d_%dx%d.yuv", filename, current_channel,
						buf, pyramid_cap->width, pyramid_cap->height);
					if (fd_pyramid[buf] < 0) {
						fd_pyramid[buf] = amba_transfer_open(pyramid_file,
							transfer_method, port++);
					}
					if (fd_pyramid[buf] < 0) {
						printf("Cannot open file [%s].\n", pyramid_file);
						continue;
					}
				}

				yuv_buffer_size = pyramid_cap->pitch *
					ROUND_UP(pyramid_cap->height, 16) * 2;
				if (!pyramid_manual_feed && alloc_gdma_dst_buf(yuv_buffer_size)) {
					rval = -1;
					break;
				}

				if (verbose) {
					gettimeofday(&curr, NULL);
					pre = curr;
				}

				if (!pyramid_manual_feed) {
					gdma_copy.src_offset = pyramid_cap->y_addr_offset;
					gdma_copy.dst_offset = 0;
					gdma_copy.src_pitch = pyramid_cap->pitch;
					gdma_copy.dst_pitch = pyramid_cap->width;
					gdma_copy.width = pyramid_cap->width;
					gdma_copy.height = pyramid_cap->height;
					if (decode_mode) {
						gdma_copy.src_mmap_type = IAV_PART_PYRAMID_POOL;
					} else {
						gdma_copy.src_mmap_type = IAV_PART_DSP;
					}
					gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
					if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
						perror("IAV_IOC_GDMA_COPY");
					}

					gdma_copy.src_offset = pyramid_cap->uv_addr_offset;
					gdma_copy.dst_offset = pyramid_cap->width * pyramid_cap->height;
					gdma_copy.src_pitch = pyramid_cap->pitch;
					gdma_copy.dst_pitch = pyramid_cap->width;
					gdma_copy.width = pyramid_cap->width;
					gdma_copy.height = pyramid_cap->height;
					if (decode_mode) {
						gdma_copy.src_mmap_type = IAV_PART_PYRAMID_POOL;
					} else {
						gdma_copy.src_mmap_type = IAV_PART_DSP;
					}
					gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
					if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
						perror("IAV_IOC_GDMA_COPY");
					}

					if (verbose) {
						gettimeofday(&curr, NULL);
						printf("12. GDMA copy [%06ld us].\n", 1000000 *
							(curr.tv_sec - pre.tv_sec) + (curr.tv_usec - pre.tv_usec));
					}
				}

				if (save_yuv_data(fd_pyramid[buf], buf, pyramid_cap, luma, chroma) < 0) {
					printf("Failed to save Pyramid data of buf [%d].\n", buf);
					rval = -1;
					break;
				}

				curr_format = get_yuv_format(yuv_format, pyramid_cap);
				if (pyramid_cap->format == IAV_YUV_FORMAT_YUV420) {
					switch (curr_format) {
					case YUV420_YV12:
						sprintf(format, "YV12");
						break;
					case YUV420_NV12:
						sprintf(format, "NV12");
						break;
					case YUV420_IYUV:
						sprintf(format, "IYUV");
						break;
					case YUV444:
						sprintf(format, "YUV444");
						break;
					default:
						sprintf(format, "IYUV");
						break;
					}
				} else {
					sprintf(format, "Unknown [%d]", pyramid_cap->format);
				}
				printf("Capture_Pyramid_buffer: Pyramid layer %d for chan %d "
					"resolution %dx%d in %s format\n", buf, current_channel,
					pyramid_cap->width, pyramid_cap->height, format);

				if (gdma_dst_buf_pid) {
					free_gdma_dst_buf();
				}
			}
			if (rval < 0) {
				break;
			}

			if (pyramid_manual_feed) {
				if (ioctl(fd_iav, IAV_IOC_RELEASE_PYRAMID_BUF, &query_desc.arg.pyramid) < 0) {
					perror("IAV_IOC_RELEASE_PYRAMID_BUF");
					rval = -1;
					break;
				}
			}

			++i;
		}
	} while (0);

	if (gdma_dst_buf_pid) {
		free_gdma_dst_buf();
	}
	if (luma) {
		free(luma);
	}
	if (chroma) {
		free(chroma);
	}
	return rval;
}

static int save_me_luma_buffer(int buf_id, u8* output, struct iav_me_cap *me_cap)
{
	int i;
	u8 *in = NULL;
	u8 *out = NULL;
	u8 *base = NULL;

	if (me_cap->pitch < me_cap->width) {
		printf("pitch size smaller than width!\n");
		return -1;
	}

	base = get_buffer_base(buf_id, 1);

	if (base == NULL) {
		printf("Invalid buffer address for buffer %d ME!"
			" Map ME buffer from DSP first.\n", buf_id);
		return -1;
	}

	if (me_cap->pitch == me_cap->width) {
		memcpy(output, base + me_cap->data_addr_offset,
			me_cap->width * me_cap->height);
	} else {
		in = base + me_cap->data_addr_offset;
		out = output;
		for (i = 0; i < me_cap->height; i++) {	//row
			memcpy(out, in, me_cap->width);
			in += me_cap->pitch;
			out += me_cap->width;
		}
	}

	return 0;
}

static int save_me_data(int fd, int buf_id, struct iav_me_cap *me_cap,
	u8 *y_buf, u8 *uv_buf)
{
	static u32 pts_prev = 0, pts = 0;

	save_me_luma_buffer(buf_id, y_buf, me_cap);

	if (amba_transfer_write(fd, y_buf, me_cap->width * me_cap->height,
		transfer_method) < 0) {
		perror("Failed to save ME data into file !\n");
		return -1;
	}

	if (amba_transfer_write(fd, uv_buf, me_cap->width * me_cap->height / 2,
		transfer_method) < 0) {
		perror("Failed to save ME data into file !\n");
		return -1;
	}

	if (verbose) {
		pts = me_cap->mono_pts;
		printf("BUF [%d] 0x%08x, pitch %d, %dx%d, seqnum [%d], PTS [%d-%d].\n",
			buf_id, (u32)me_cap->data_addr_offset, me_cap->pitch,
			me_cap->width, me_cap->height, me_cap->seq_num,
			pts, (pts - pts_prev));
		pts_prev = pts;
	}

	return 0;
}

static int query_me(int buf_map, int count, int is_me1)
{
	int i = 0, buf;
	int non_stop = 0;
	struct iav_querydesc query_desc;
	struct iav_canvasdesc *canvas_desc;
	struct iav_me_cap *me_cap;
	static int pts_prev = 0, pts = 0;

	if (count == 0) {
		non_stop = 1;
	}

	canvas_desc = &query_desc.arg.canvas;
	if (is_me1) {
		me_cap = &canvas_desc->me1;
	} else {
		me_cap = &canvas_desc->me0;
	}

	while ((i < count || non_stop) && !quit_capture) {
		for (buf = 0; buf < G_canvas_num; ++buf) {
			if (buf_map & (1 << buf)) {
				memset(&query_desc, 0, sizeof(query_desc));
				query_desc.qid = IAV_DESC_CANVAS;
				query_desc.arg.canvas.canvas_id = buf;
				if (!non_block_read) {
					canvas_desc->non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
				} else {
					canvas_desc->non_block_flag |= IAV_BUFCAP_NONBLOCK;
				}

				gettimeofday(&curr, NULL);
				pre = curr;

				if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
					if (errno == EINTR) {
						continue;		/* back to for() */
					} else {
						perror("IAV_IOC_QUERY_DESC");
						goto query_me_error_exit;
					}
				}

				gettimeofday(&curr, NULL);
				printf("Query ME%d DESC [%06ld us].\n", is_me1, 1000000 *
					(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));

				if (me_cap->data_addr_offset == 0) {
					printf("ME buffer [%d] address is NULL! Skip to next!\n", buf);
					continue;
				}

				pts = me_cap->mono_pts;
				printf("Me_buffer: resolution %dx%d with Luma only,"
					"seq num[%u] PTS [%u-%u] idsp pts[%u].\n",
					me_cap->width, me_cap->height, me_cap->seq_num,
					pts, (pts - pts_prev), me_cap->dsp_pts);
				pts_prev = pts;
			}
		}
		++i;
	}

	return 0;

query_me_error_exit:
	return -1;
}

static int capture_me(int buf_map, int count, int is_me1)
{
	int i, buf, save[IAV_SRCBUF_NUM];
	int write_flag[IAV_SRCBUF_NUM];
	char me_file[320];
	int non_stop = 0;
	u8 * luma = NULL;
	u8 * chroma = NULL;
	struct iav_querydesc query_desc;
	struct iav_canvasdesc *canvas_desc;
	struct iav_me_cap *me_cap;
	struct iav_me_cap me_cap_cache;

	luma = (u8*)malloc(MAX_ME_BUFFER_SIZE);
	if (luma == NULL) {
		printf("Not enough memory for ME buffer capture !\n");
		goto me_error_exit;
	}

	//clear UV to be B/W mode, UV data is not useful for motion detection,
	//just fill UV data to make YUV to be YV12 format, so that it can play in YUV player
	chroma = (u8*)malloc(MAX_ME_BUFFER_SIZE);
	if (chroma == NULL) {
		printf("Not enough memory for ME buffer capture !\n");
		goto me_error_exit;
	}
	memset(chroma, 0x80, MAX_ME_BUFFER_SIZE);
	memset(save, 0, sizeof(save));
	memset(write_flag, 0, sizeof(write_flag));
	memset(luma, 1, MAX_ME_BUFFER_SIZE);

	if (count == 0) {
		non_stop = 1;
	}

	canvas_desc = &query_desc.arg.canvas;
	if (is_me1) {
		me_cap = &canvas_desc->me1;
	} else {
		me_cap = &canvas_desc->me0;
	}

	i = 0;
	while ((i < count || non_stop) && !quit_capture) {
		for (buf = 0; buf < G_canvas_num; ++buf) {
			if (buf_map & (1 << buf)) {
				memset(&query_desc, 0, sizeof(query_desc));
				query_desc.qid = IAV_DESC_CANVAS;
				query_desc.arg.canvas.canvas_id = buf;
				if (!non_block_read) {
					canvas_desc->non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
				} else {
					canvas_desc->non_block_flag |= IAV_BUFCAP_NONBLOCK;
				}
				if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
					if (errno == EINTR) {
						continue;		/* back to for() */
					} else {
						perror("IAV_IOC_QUERY_DESC");
						goto me_error_exit;
					}
				}

				if (fd_me[buf] < 0) {
					memset(me_file, 0, sizeof(me_file));
					if (!me_cap->width) {
						continue;
					}
					sprintf(me_file, "%s_canvas%d_me_%dx%d.yuv", filename, buf,
						me_cap->width, me_cap->height);
					if (fd_me[buf] < 0) {
						fd_me[buf] = amba_transfer_open(me_file,
							transfer_method, port++);
					}
					if (fd_me[buf] < 0) {
						printf("Cannot open file [%s].\n", me_file);
						continue;
					}
				}

				if (me_cap->data_addr_offset == 0) {
					printf("ME buffer [%d] address is NULL! Skip to next!\n", buf);
					continue;
				}
				if (delay_frame_cap_data) {
					if (write_flag[buf] == 0) {
						write_flag[buf] = 1;
						me_cap_cache = *me_cap;
					} else {
						write_flag[buf] = 0;
						if (save_me_data(fd_me[buf], buf, &me_cap_cache, luma, chroma) < 0) {
							printf("Failed to save ME data of buf [%d].\n", buf);
							goto me_error_exit;
						}
					}
				} else {
					if (save_me_data(fd_me[buf], buf, me_cap, luma, chroma) < 0) {
						printf("Failed to save ME data of buf [%d].\n", buf);
						goto me_error_exit;
					}
				}

				if (save[buf] == 0) {
					save[buf] = 1;
					printf("Delay %d frame capture me data.\n", delay_frame_cap_data);
					printf("Save_me_buffer: resolution %dx%d with Luma only.\n",
						me_cap->width, me_cap->height);
				}
			}
		}
		++i;
	}

	free(luma);
	free(chroma);
	return 0;

me_error_exit:
	if (luma)
		free(luma);
	if (chroma)
		free(chroma);
	return -1;
}

static int query_raw(int count)
{
	struct iav_rawbufdesc *raw_desc;
	struct iav_querydesc query_desc;
	struct iav_system_resource resource;
	static int pts_prev = 0, pts = 0;
	int i = 0, non_stop = 0;

	if (count == 0) {
		non_stop = 1;
	}

	while ((i < count || non_stop) && !quit_capture) {

		memset(&resource, 0, sizeof(resource));
		resource.encode_mode = DSP_CURRENT_MODE;
		if (ioctl(fd_iav, IAV_IOC_GET_SYSTEM_RESOURCE, &resource) < 0) {
			perror("IAV_IOC_GET_SYSTEM_RESOURCE");
			goto query_raw_error_exit;
		}
		memset(&query_desc, 0, sizeof(query_desc));
		query_desc.qid = IAV_DESC_RAW;
		raw_desc = &query_desc.arg.raw;
		raw_desc->vin_id = vinc_id;

		gettimeofday(&curr, NULL);
		pre = curr;

		if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
			if (errno == EINTR) {
				// skip to do nothing
			} else {
				perror("IAV_IOC_QUERY_DESC");
				goto query_raw_error_exit;
			}
		}

		gettimeofday(&curr, NULL);
		printf("Query RAW DESC [%06ld us].\n", 1000000 *
			(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));

		if (!raw_desc->pitch || !raw_desc->height || !raw_desc->width) {
			printf("Raw data resolution %ux%u with pitch %u is incorrect!\n",
				raw_desc->width, raw_desc->height, raw_desc->pitch);
			goto query_raw_error_exit;
		}

		pts = raw_desc->mono_pts;

		printf("Raw data resolution %u x %u with pitch %u, PTS [%u-%u]..\n",
		raw_desc->width, raw_desc->height, raw_desc->pitch, pts, (pts - pts_prev));
		pts_prev = pts;

		i++;
	}

	return 0;

query_raw_error_exit:

	return -1;
}

static int capture_raw(int count)
{
	struct iav_rawbufdesc *raw_desc;
	struct iav_querydesc query_desc;
	struct iav_system_resource resource;
	u32 buffer_size;
	char raw_file[NAME_SIZE];
	struct iav_gdma_copy gdma_copy = {0};
	struct vindev_video_info video_info;
	struct vindev_devinfo vin_info;
	int i;
	struct iav_chan_cfg* chan_cfg = NULL;
	int rval = 0;

	if (count <= 0) {
		count = 1;
	}

	while (count > 0) {
		memset(&query_desc, 0, sizeof(query_desc));
		query_desc.qid = IAV_DESC_RAW;
		raw_desc = &query_desc.arg.raw;
		raw_desc->vin_id = vinc_id;
		if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
			if (errno == EINTR) {
				// skip to do nothing
			} else {
				perror("IAV_IOC_QUERY_DESC");
				rval = -1;
				break;
			}
		}

		if (!raw_desc->pitch || !raw_desc->height || !raw_desc->width) {
			printf("Raw data resolution %ux%u with pitch %u is incorrect!\n",
				raw_desc->width, raw_desc->height, raw_desc->pitch);
			rval = -1;
			break;
		}

		if (fd_raw < 0) {
			/* create fd */
			memset(raw_file, 0, sizeof(raw_file));
			sprintf(raw_file, "%s_raw_%dx%d_%d", filename, raw_desc->width,
				raw_desc->height, raw_desc->pitch);

			memset(&resource, 0, sizeof(resource));
			resource.encode_mode = DSP_CURRENT_MODE;
			if (ioctl(fd_iav, IAV_IOC_GET_SYSTEM_RESOURCE, &resource) < 0) {
				perror("IAV_IOC_GET_SYSTEM_RESOURCE");
				rval = -1;
				break;
			}

			/* find first channel which is related with selected vin */
			for (i = 0; i < resource.chan_num; i++) {
				vin_info.vsrc_id = resource.chan_cfg[i].vsrc_id;
				if (ioctl(fd_iav, IAV_IOC_VIN_GET_DEVINFO, &vin_info) < 0) {
					perror("IAV_IOC_VIN_GET_DEVINFO error\n");
					break;
				}
				if (vin_info.vinc_id== vinc_id) {
					chan_cfg = &resource.chan_cfg[i];
					break;
				}
			}
			if (!chan_cfg) {
				printf("Could not find related channel.\n");
				goto CAPTURE_RAW_EXIT;
				rval = -1;
			}

			if (chan_cfg->packing_mode_enable) {
				memset(&video_info, 0, sizeof(video_info));
				video_info.vsrc_id = chan_cfg->vsrc_id;
				video_info.info.mode = AMBA_VIDEO_MODE_CURRENT;
				if (ioctl(fd_iav, IAV_IOC_VIN_GET_VIDEOINFO, &video_info) < 0) {
					perror("IAV_IOC_VIN_GET_VIDEOINFO");
					rval = -1;
					break;
				}

				sprintf(raw_file + strlen(raw_file), "_pck_%dbits",
					video_info.info.bits);
			}

			if (!chan_cfg->raw_capture) {
				sprintf(raw_file + strlen(raw_file), "_cpr");
			}

			sprintf(raw_file + strlen(raw_file), ".raw");

			if (fd_raw < 0) {
				fd_raw = amba_transfer_open(raw_file,
					transfer_method, port++);
			}
			if (fd_raw < 0) {
				printf("Cannot open file [%s].\n", raw_file);
				rval = -1;
				break;
			}
		}

		buffer_size = raw_desc->pitch * raw_desc->height;
		if (alloc_gdma_dst_buf(buffer_size) < 0) {
			rval = -1;
			break;
		}

		gdma_copy.src_offset = raw_desc->raw_addr_offset;
		gdma_copy.dst_offset = 0;
		gdma_copy.src_pitch = raw_desc->pitch;
		gdma_copy.dst_pitch = raw_desc->pitch;
		gdma_copy.width = raw_desc->pitch;
		gdma_copy.height = raw_desc->height;
		gdma_copy.src_mmap_type = IAV_PART_DSP;
		gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
		if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
			perror("IAV_IOC_GDMA_COPY");
			rval = -1;
			break;
		}

		if (amba_transfer_write(fd_raw, gdma_dst_buf, buffer_size, transfer_method) < 0) {
			perror("Failed to save RAW data into file !\n");
			rval = -1;
			break;
		}

		count--;
	}

	if (rval >= 0) {
		printf("save raw buffer done!\n");
		printf("Raw data resolution %u x %u with pitch %u.\n",
			raw_desc->width, raw_desc->height, raw_desc->pitch);
	}

CAPTURE_RAW_EXIT:
	if (gdma_dst_buf_pid) {
		free_gdma_dst_buf();
	}
	if (fd_raw >= 0) {
		amba_transfer_close(fd_raw, transfer_method);
		fd_raw = -1;
	}
	return rval;
}

static int check_state(void)
{
	int state;
	if (ioctl(fd_iav, IAV_IOC_GET_IAV_STATE, &state) < 0) {
		perror("IAV_IOC_GET_IAV_STATE");
		exit(2);
	}

	if ((state != IAV_STATE_PREVIEW) && (state != IAV_STATE_ENCODING) &&
		(state != IAV_STATE_DECODING)) {
		printf("IAV is not in preview / encoding /decoding state, cannot get yuv buf!\n");
		return -1;
	}

	if (state == IAV_STATE_DECODING) {
		decode_mode = 1;
	}

	return 0;
}

static int get_resource_info(void)
{
	struct iav_system_resource resource;
	struct iav_pyramid_cfg pyramid_cfg;
	u8 i, frame_rate;

	// system resource
	memset(&resource, 0, sizeof(struct iav_system_resource));
	resource.encode_mode = DSP_CURRENT_MODE;
	if (ioctl(fd_iav, IAV_IOC_GET_SYSTEM_RESOURCE, &resource) < 0) {
		perror("IAV_IOC_GET_SYSTEM_RESOURCE\n");
		return -1;
	}

	G_multi_vin_num = resource.chan_num;
	G_canvas_num = resource.canvas_num;

	for (i = 0; i < IAV_MAX_CANVAS_BUF_NUM; i++) {
		frame_rate = resource.canvas_cfg[i].frame_rate;
		if (frame_rate != 0) {
			pts_intval[i] = HWTIMER_OUTPUT_FREQ / frame_rate;
		}
	}

	if (current_channel < G_multi_vin_num) {
		memset(&pyramid_cfg, 0, sizeof(struct iav_pyramid_cfg));
		pyramid_cfg.chan_id = current_channel;
		if (ioctl(fd_iav, IAV_IOC_GET_PYRAMID_CFG, &pyramid_cfg) < 0) {
			perror("IAV_IOC_GET_PYRAMID_CFG\n");
			return -1;
		}
		pyramid_manual_feed = pyramid_cfg.manual_feed;
	} else {
		printf("The channel_id[%d] cannot excess channel_num[%d]\n",
			current_channel, G_multi_vin_num);
		return -1;
	}

	return 0;
}

static int get_yuv_data(int buf_id, struct iav_yuv_cap *yuv_cap, u8 *luma, u8 *chroma, u8* buffer)
{
	static int pts_prev = 0, pts = 0;
	int luma_size, chroma_size;
	u8 *base = NULL;
	u8 * luma_addr = NULL, *chroma_addr = NULL;
	int format = get_yuv_format(yuv_format, yuv_cap);
	int ret = 0;
	
	luma_size = yuv_cap->width * yuv_cap->height;
	chroma_size = luma_size >> 1;

	base = get_buffer_base(buf_id, 0);
	luma_addr = base + yuv_cap->y_addr_offset;
	chroma_addr = base + yuv_cap->uv_addr_offset;

	if (save_yuv_luma_buffer(buf_id, luma_addr, luma, yuv_cap) < 0) {
		perror("Failed to save luma data into buffer !\n");
		return -1;
	}
	luma_addr = luma;
	if (verbose) {
		gettimeofday(&curr, NULL);
		printf("22. Save Y [%06ld us].\n", (curr.tv_sec - pre.tv_sec) *
			1000000 + (curr.tv_usec - pre.tv_usec));
		pre = curr;
	}
	if (save_yuv_chroma_buffer(buf_id, chroma_addr, chroma, yuv_cap, 0) < 0) {
		perror("Failed to save chroma data into buffer !\n");
		return -1;
	}
	chroma_addr = chroma;


//	printf("luma_addr:0x%x, luma_size:%d, chroma_addr:0x%x, chroma_size:%d\n", luma_addr, luma_size, chroma_addr, chroma_size);
	if (verbose) {
        gettimeofday(&curr, NULL);
        pre = curr;
    }
	memcpy(buffer, luma_addr, luma_size);
	memcpy(buffer + luma_size, chroma_addr, chroma_size);
	if (verbose) {
        gettimeofday(&curr, NULL);
        printf("44. Copy yuv [%06ld us].\n", (curr.tv_sec - pre.tv_sec) *
            1000000 + (curr.tv_usec - pre.tv_usec));
    }

	if (verbose) {
		pts = yuv_cap->mono_pts;
		printf("BUF Y 0x%08x, UV 0x%08x, pitch %u, %ux%u = Seqnum[%u], "
			"PTS [%u-%u].\n", (u32)yuv_cap->y_addr_offset,
			(u32)yuv_cap->uv_addr_offset, yuv_cap->pitch, yuv_cap->width,
			yuv_cap->height, yuv_cap->seq_num, pts, (pts - pts_prev));
		pts_prev = pts;
	}

	return ret;
}

static int capture_yuv_data(int buffer_id, u8* buffer)
{
	u8 *luma = NULL, *chroma = NULL;
	int i = 0;
	int yuv_buffer_size = 0;
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *yuv_cap;
	struct iav_gdma_copy gdma_copy = {0};
	int rval = 0;

	luma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
	if (luma == NULL) {
		printf("Not enough memory for preview capture !\n");
		return -1;
	}
	chroma = (u8*)malloc(MAX_YUV_BUFFER_SIZE);
	if (chroma == NULL) {
		printf("Not enough memory for preview capture !\n");
		return -1;
	}
	memset(luma, 1, MAX_YUV_BUFFER_SIZE);
	memset(chroma, 1, MAX_YUV_BUFFER_SIZE);

	/* query */
	memset(&query_desc, 0, sizeof(query_desc));
	query_desc.qid = IAV_DESC_CANVAS;
	query_desc.arg.canvas.canvas_id = buffer_id;
	if (!non_block_read) {
		query_desc.arg.canvas.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;
	} else {
		query_desc.arg.canvas.non_block_flag |= IAV_BUFCAP_NONBLOCK;
	}

	if (verbose) {
		gettimeofday(&curr, NULL);
		pre = curr;
	}
	if (ioctl(fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
		perror("IAV_IOC_QUERY_DESC");
		return -1;
	}
	if (verbose) {
		gettimeofday(&curr, NULL);
		printf("11. Query DESC [%06ld us].\n", 1000000 *
			(curr.tv_sec - pre.tv_sec)  + (curr.tv_usec - pre.tv_usec));
	}
	yuv_cap = &query_desc.arg.canvas.yuv;
	if ((yuv_cap->y_addr_offset == 0) || (yuv_cap->uv_addr_offset == 0)) {
		printf("YUV buffer [%d] address is NULL! Skip to next!\n", buffer_id);
		return -1;
	}

	yuv_buffer_size = yuv_cap->pitch * ROUND_UP(yuv_cap->height, 16) * 2;
	if (alloc_gdma_dst_buf(yuv_buffer_size)) {
		return -1;
	}

	if (verbose) {
		gettimeofday(&curr, NULL);
		pre = curr;
	}

	gdma_copy.src_offset = yuv_cap->y_addr_offset;
	gdma_copy.dst_offset = 0;
	gdma_copy.src_pitch = yuv_cap->pitch;
	gdma_copy.dst_pitch = yuv_cap->width;
	gdma_copy.width = yuv_cap->width;
	gdma_copy.height = yuv_cap->height;
	gdma_copy.src_mmap_type = IAV_PART_DSP;
	gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
	if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
		perror("IAV_IOC_GDMA_COPY");
	}

	gdma_copy.src_offset = yuv_cap->uv_addr_offset;
	gdma_copy.dst_offset = yuv_cap->width * yuv_cap->height;
	gdma_copy.src_pitch = yuv_cap->pitch;
	gdma_copy.dst_pitch = yuv_cap->width;
	gdma_copy.width = yuv_cap->width;
	gdma_copy.height = yuv_cap->height;
	gdma_copy.src_mmap_type = IAV_PART_DSP;
	gdma_copy.dst_mmap_type = gdma_dst_buf_pid;
	if (ioctl(fd_iav, IAV_IOC_GDMA_COPY, &gdma_copy) < 0) {
		perror("IAV_IOC_GDMA_COPY");
	}

	if (verbose) {
		gettimeofday(&curr, NULL);
		printf("12. GDMA copy [%06ld us].\n", 1000000 *
			(curr.tv_sec - pre.tv_sec) + (curr.tv_usec - pre.tv_usec));
	}

	if (get_yuv_data(buffer_id, yuv_cap, luma, chroma, buffer) < 0) {
		LOG(ERROR) << "Failed to save YUV data of buf " << buffer;
		return -1;
	}

	if (gdma_dst_buf_pid) {
		free_gdma_dst_buf();
	}
	if (luma) {
		free(luma);
	}
	if (chroma) {
		free(chroma);
	}

	return rval;
}

static void *run_camera_pthread(void* data)
{
	uint64_t start_time = 0;
	int buffer_id = 3;
	int policy = -1;
    struct sched_param param;
    pthread_getschedparam(pthread_self(),&policy,&param);
    if(policy == SCHED_OTHER)
		LOG(WARNING) << "SCHED_OTHER";
    if(policy == SCHED_RR)
		LOG(WARNING) << "SCHED_RR";
    if(policy==SCHED_FIFO)
		LOG(WARNING) << "SCHED_FIFO";
	LOG(WARNING) << "sched_priority:" << param.sched_priority;
	prctl(PR_SET_NAME, "camera_pthread");
	while(run_camera) {
		start_time = gettimeus();
		pthread_mutex_lock(&image_buffer.lock);  
		if ((image_buffer.writepos + 1) % IMAGE_BUFFER_SIZE == image_buffer.readpos)  
		{  
#if defined(ONLY_SAVE_DATA)
			struct timeval now;
    		struct timespec outtime;
			gettimeofday(&now, NULL);
    		outtime.tv_sec = now.tv_sec + 1;
    		outtime.tv_nsec = now.tv_usec * 1000;
			pthread_cond_timedwait(&image_buffer.notfull, &image_buffer.lock, &outtime);
#else
			pthread_cond_wait(&image_buffer.notfull, &image_buffer.lock);
#endif  
		}
		// std::cout << "1111111111111111111" << std::endl;
		memset(image_buffer.buffer[image_buffer.writepos], 0, IMAGE_YUV_SIZE * sizeof(u8));
		start_time = gettimeus();
		if(capture_yuv_data(buffer_id, image_buffer.buffer[image_buffer.writepos]) < 0)
		{
			LOG(ERROR) << "capture yuv data fail!";
		}
		else
		{
			image_buffer.buffer_stamp[image_buffer.writepos] = get_time_stamp();
			image_buffer.writepos++;
		}
		// LOG(WARNING) << "get yuv cost time:" <<  (gettimeus() - start_time)/1000.0  << "ms";
		if (image_buffer.writepos >= IMAGE_BUFFER_SIZE)  
			image_buffer.writepos = 0;  
		pthread_cond_signal(&image_buffer.notempty);  
		pthread_mutex_unlock(&image_buffer.lock); 
		// LOG(WARNING) << "get image pthread all cost time:" <<  (gettimeus() - start_time)/1000.0  << "ms";
	}
	run_camera = 0;
	LOG(WARNING) << "Camera thread quit.";
	return NULL;
}

ImageAcquisition::ImageAcquisition()
{
	run_camera = 0;
	pthread_id = 0;
	LOG(WARNING) << IMAGE_BUFFER_SIZE;
}

ImageAcquisition::~ImageAcquisition()
{
	if(run_camera > 0)
	{
		stop();
	}
	pthread_attr_destroy(&pthread_attr);
	pthread_mutex_destroy(&image_buffer.lock);
    pthread_cond_destroy(&image_buffer.notempty);
    pthread_cond_destroy(&image_buffer.notfull);
    quit_capture = 1;
	if(fd_iav >= 0)
	{
		close(fd_iav);
	}
	LOG(WARNING) << "~ImageAcquisition()";
}

int ImageAcquisition::open_camera()
{
	u32 buffer_map = 0;
    // open the device
	if ((fd_iav = open("/dev/iav", O_RDWR, 0)) < 0) 
    {
        LOG(ERROR) << "open /dev/iav fail!";
		return -1;
	}
    //check iav state
	if (check_state() < 0)
    {
        LOG(ERROR) << "check_state fail!";
        return -1;
    }
    if (map_buffer() < 0)
    {
        LOG(ERROR) << "map_buffer fail!";
        return -1;
    }

	for (int i = 0; i < IAV_MAX_CANVAS_BUF_NUM; ++i) {
		fd_yuv[i] = -1;
		fd_me[i] = -1;
	}
	for (int i = 0; i < IAV_MAX_PYRAMID_LAYERS; ++i) {
		fd_pyramid[i] = -1;
	}

	current_buffer = 3;
	VERIFY_BUFFERID(current_buffer);
	capture_select = CAPTURE_PREVIEW_BUFFER;
	yuv_buffer_id |= (1 << current_buffer);
	strcpy(filename, "imx416");
	yuv_format = YUV420_IYUV;

	pthread_mutex_init(&image_buffer.lock, NULL);  
    pthread_cond_init(&image_buffer.notempty, NULL);  
    pthread_cond_init(&image_buffer.notfull, NULL);  
    image_buffer.readpos = 0;  
    image_buffer.writepos = 0;
	LOG(INFO) << "Camera open success!";
    return 0;
}

int ImageAcquisition::start()
{
    int ret = 0;
	struct sched_param param;
	if(fd_iav >= 0)
	{
		run_camera = 1;
		image_buffer.readpos = 0;  
		image_buffer.writepos = 0;
		pthread_attr_init(&pthread_attr);
		param.sched_priority = 51;
		pthread_attr_setschedpolicy(&pthread_attr, SCHED_RR);
		pthread_attr_setschedparam(&pthread_attr, &param);
		pthread_attr_setinheritsched(&pthread_attr, PTHREAD_EXPLICIT_SCHED);
		ret = pthread_create(&pthread_id, &pthread_attr, run_camera_pthread, NULL);
		if(ret < 0)
		{
			run_camera = 0;
			LOG(ERROR) << "satrt camera pthread fail!";
		}
		LOG(INFO) << "satrt camera pthread success!";
	}
	else
	{
		LOG(ERROR) << "camera device not open!" << fd_iav;
	}
	return ret;
}

int ImageAcquisition::stop()
{
	int ret = 0;
	run_camera = 0;
	if (pthread_id > 0) {
		pthread_cond_signal(&image_buffer.notfull);
		pthread_cond_signal(&image_buffer.notempty);  
		pthread_mutex_unlock(&image_buffer.lock);
		pthread_join(pthread_id, NULL);
		pthread_id = 0;
	}
	pthread_attr_destroy(&pthread_attr);
	LOG(WARNING) << "stop camera success";
	return ret;
}

void ImageAcquisition::get_image(cv::Mat &src_image)
{
	if(pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
		cv::Mat yuvImg(IMAGE_HEIGHT + IMAGE_HEIGHT / 2, IMAGE_WIDTH, CV_8UC1, image_buffer.buffer[image_buffer.readpos]);
		cv::cvtColor(yuvImg, src_image, cv::COLOR_YUV2BGR_IYUV);
		image_buffer.readpos++;  
		if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}

void ImageAcquisition::get_image(cv::Mat &src_image, long *stamp)
{
	if(pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
		cv::Mat yuvImg(IMAGE_HEIGHT + IMAGE_HEIGHT / 2, IMAGE_WIDTH, CV_8UC1, image_buffer.buffer[image_buffer.readpos]);
		cv::cvtColor(yuvImg, src_image, cv::COLOR_YUV2BGR_IYUV);
		*stamp = image_buffer.buffer_stamp[image_buffer.readpos];
		image_buffer.readpos++;  
		if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}

void ImageAcquisition::get_yuv(unsigned char* addr)
{
	if(pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
		memcpy(addr, image_buffer.buffer[image_buffer.readpos], IMAGE_YUV_SIZE * sizeof(unsigned char));
		image_buffer.readpos++;  
		if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}

void ImageAcquisition::get_yuv(unsigned char* addr, long *stamp)
{
	if(pthread_id > 0)
	{
		pthread_mutex_lock(&image_buffer.lock);  
		if (image_buffer.writepos == image_buffer.readpos)  
		{  
			pthread_cond_wait(&image_buffer.notempty, &image_buffer.lock);  
		}
		memcpy(addr, image_buffer.buffer[image_buffer.readpos], IMAGE_YUV_SIZE * sizeof(unsigned char));
		*stamp = image_buffer.buffer_stamp[image_buffer.readpos];
		image_buffer.readpos++;  
		if (image_buffer.readpos >= IMAGE_BUFFER_SIZE)  
			image_buffer.readpos = 0; 
		pthread_cond_signal(&image_buffer.notfull);  
		pthread_mutex_unlock(&image_buffer.lock);
	}
}