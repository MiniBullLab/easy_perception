#ifndef __VOUT_TOOL_H__
#define __VOUT_TOOL_H__

int set_vout_bbox(bbox_list_t *bbox_list);
int set_vout_image(ea_tensor_t *complete_img, draw_plate_list_t* draw_plate_list);
int show_vout(uint32_t dsp_pts);
int init_vout_tool(int stream_id, float x_offset,
	uint32_t highlight_sec, uint32_t clear_sec, float width_ratio,
	uint32_t draw_plate_num, uint32_t debug_en);
void deinit_vout_tool();

#endif

