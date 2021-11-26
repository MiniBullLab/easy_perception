#pragma once
#if 0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


class RC4
{
public:
    RC4();
    ~RC4();
    void rc4_input (unsigned char *pass, unsigned int keylen);
    unsigned char rc4_output_v01();
    unsigned char rc4_output();


private:
    unsigned char s[256];
    unsigned int i, j;
    int size;
    void swap(unsigned char *s, unsigned int i, unsigned int j);
};

#define BIT(x,y)    ((x>>y)&1)
#define SET_BIT(x,y)    (x|=(1<<y))
#define CLEAR_BIT(x,y)    (x&=~(1<<y))


#define FLASH_HEADER_START_ADDR      524288  // 0X80000

typedef unsigned char BYTE;
typedef unsigned long DWORD;

struct header_struct
{
    unsigned short usVersion;
    char* sensor_type;
    unsigned short usSupportFrequency;
    unsigned short usLensType;
    unsigned short usVCSEL;
    unsigned short usPlatform;
    unsigned short usPCB;
    char* encrypt_key;
    int data_size;
    int data_offset;
};

#if 0
enum {
	SENSOR_IMX316 = 0,
	SENSOR_IMX456,
	SENSOR_MAX,
};
#endif

#define FREQUENCY_60HZ_SHIFT			0
#define FREQUENCY_100HZ_SHIFT			1
#define FREQUENCY_37_5HZ_SHIFT		2
#define FREQUENCY_MAX



#define SIZEOF_HEADER_VERSION			2
#define SIZEOF_SENSOR_TYPE				10
#define SIZEOF_HEADER_SIZE				2
#define SIZEOF_LENS_TYPE				2
#define SIZEOF_VCEL						2
#define SIZEOF_PLATEFORM				2
#define SIZEOF_PCB						2
#define SIZEOF_YEAR						2
#define SIZEOF_MONTH					2
#define SIZEOF_DAY						2
#define SIZEOF_TIME						4
#define SIZEOF_SERIAL_NUMBER			16
#define SIZEOF_RESEARVED				16
#define SIZEOF_ENCRYPTION_KEY			16
#define SIZEOF_DATA_VERSION			6
#define SIZEOF_DATA_OFFSET				4
#define SIZEOF_DATA_SIZE				4
#define SIZEOF_DATA_CHECKSUM			2


#define OFFSETOF_HEADER_VERSION			(0)
#define OFFSETOF_SENSOR_TYPE  				(OFFSETOF_HEADER_VERSION		+SIZEOF_HEADER_VERSION)
#define OFFSETOF_HEADER_SIZE				(OFFSETOF_SENSOR_TYPE			+SIZEOF_SENSOR_TYPE)
#define OFFSETOF_LENS_TYPE					(OFFSETOF_HEADER_SIZE			+SIZEOF_HEADER_SIZE)
#define OFFSETOF_VCEL						(OFFSETOF_LENS_TYPE				+SIZEOF_LENS_TYPE)
#define OFFSETOF_PLATEFORM				(OFFSETOF_VCEL					+SIZEOF_VCEL)
#define OFFSETOF_PCB						(OFFSETOF_PLATEFORM				+SIZEOF_PLATEFORM)
#define OFFSETOF_YEAR						(OFFSETOF_PCB					+SIZEOF_PCB)
#define OFFSETOF_MONTH					(OFFSETOF_YEAR					+SIZEOF_YEAR)
#define OFFSETOF_DAY						(OFFSETOF_MONTH				+SIZEOF_MONTH)
#define OFFSETOF_TIME						(OFFSETOF_DAY					+SIZEOF_DAY)
#define OFFSETOF_SERIAL_NUMBER			(OFFSETOF_TIME					+SIZEOF_TIME)
#define OFFSETOF_RESEARVED				(OFFSETOF_SERIAL_NUMBER			+SIZEOF_SERIAL_NUMBER)
#define OFFSETOF_ENCRYPTION_KEY			(OFFSETOF_RESEARVED				+SIZEOF_RESEARVED)
#define OFFSETOF_DATA_VERSION				(OFFSETOF_ENCRYPTION_KEY		+SIZEOF_ENCRYPTION_KEY)
#define OFFSETOF_DATA_OFFSET				(OFFSETOF_DATA_VERSION			+SIZEOF_DATA_VERSION)
#define OFFSETOF_DATA_SIZE					(OFFSETOF_DATA_OFFSET			+SIZEOF_DATA_OFFSET)
#define OFFSETOF_DATA_CHECKSUM			(OFFSETOF_DATA_SIZE				+SIZEOF_DATA_SIZE)



int UploadCalibration_SunplusUVC(BYTE *buffer, const DWORD dwBufferLen, const DWORD dwAddress);

#ifdef __AMBA_FLASH__TEST
#include "amba_cvwarp.h"
int UploadCalibration_AmbaSPI(amba_cvwarp_api *cvwarp, BYTE *buffer, const DWORD dwBufferLen, const DWORD dwAddress);
#endif

//int read_data_from_flash(int start_sector,unsigned char* pdata, int size);
#else
class RC4
{
public:
	RC4();
	~RC4();
	void rc4_input(unsigned char *pass, unsigned int keylen);
	unsigned char rc4_output_v01();
	unsigned char rc4_output();


private:
	unsigned char s[256];
	unsigned int i, j;
	int size;
	void swap(unsigned char *s, unsigned int i, unsigned int j);

};

#define FLASH_HEADER_START_ADDR      524288  // 0X80000
typedef unsigned char BYTE;
typedef unsigned long DWORD;

struct header_struct
{
	unsigned short usVersion;
	char* sensor_type;
	unsigned short usSupportFrequency;
	unsigned short usLensType;
	unsigned short usVCSEL;
	unsigned short usPlatform;
	unsigned short usPCB;
	char* encrypt_key;
	int data_size;
	int data_offset;
};

#define SIZEOF_HEADER_VERSION			2
#define SIZEOF_SENSOR_TYPE			10
#define SIZEOF_HEADER_SIZE			2
#define SIZEOF_LENS_TYPE			2
#define SIZEOF_VCEL				2
#define SIZEOF_PLATEFORM			2
#define SIZEOF_PCB				2
#define SIZEOF_YEAR				2
#define SIZEOF_MONTH				2
#define SIZEOF_DAY				2
#define SIZEOF_TIME				4
#define SIZEOF_SERIAL_NUMBER			16
#define SIZEOF_RESEARVED			16
#define SIZEOF_ENCRYPTION_KEY			16
#define SIZEOF_DATA_VERSION			6
#define SIZEOF_DATA_OFFSET			4
#define SIZEOF_DATA_SIZE			4
#define SIZEOF_DATA_CHECKSUM			2


#define OFFSETOF_HEADER_VERSION			(0)
#define OFFSETOF_SENSOR_TYPE  			(OFFSETOF_HEADER_VERSION		+SIZEOF_HEADER_VERSION)
#define OFFSETOF_HEADER_SIZE			(OFFSETOF_SENSOR_TYPE			+SIZEOF_SENSOR_TYPE)
#define OFFSETOF_LENS_TYPE			(OFFSETOF_HEADER_SIZE			+SIZEOF_HEADER_SIZE)
#define OFFSETOF_VCEL				(OFFSETOF_LENS_TYPE			+SIZEOF_LENS_TYPE)
#define OFFSETOF_PLATEFORM			(OFFSETOF_VCEL				+SIZEOF_VCEL)
#define OFFSETOF_PCB				(OFFSETOF_PLATEFORM			+SIZEOF_PLATEFORM)
#define OFFSETOF_YEAR				(OFFSETOF_PCB				+SIZEOF_PCB)
#define OFFSETOF_MONTH				(OFFSETOF_YEAR				+SIZEOF_YEAR)
#define OFFSETOF_DAY				(OFFSETOF_MONTH				+SIZEOF_MONTH)
#define OFFSETOF_TIME				(OFFSETOF_DAY				+SIZEOF_DAY)
#define OFFSETOF_SERIAL_NUMBER			(OFFSETOF_TIME				+SIZEOF_TIME)
#define OFFSETOF_RESEARVED			(OFFSETOF_SERIAL_NUMBER			+SIZEOF_SERIAL_NUMBER)
#define OFFSETOF_ENCRYPTION_KEY			(OFFSETOF_RESEARVED			+SIZEOF_RESEARVED)
#define OFFSETOF_DATA_VERSION			(OFFSETOF_ENCRYPTION_KEY		+SIZEOF_ENCRYPTION_KEY)
#define OFFSETOF_DATA_OFFSET			(OFFSETOF_DATA_VERSION			+SIZEOF_DATA_VERSION)
#define OFFSETOF_DATA_SIZE			(OFFSETOF_DATA_OFFSET			+SIZEOF_DATA_OFFSET)
#define OFFSETOF_DATA_CHECKSUM			(OFFSETOF_DATA_SIZE			+SIZEOF_DATA_SIZE)

#endif

