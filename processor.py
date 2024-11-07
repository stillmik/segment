import os
import cv2
import numpy as np
import torch
import gc
import imageio

from scipy.ndimage import binary_dilation
from model_args import segtracker_args, sam_args, aot_args
from PIL import Image
from SegTracker import SegTracker
from aot_tracker import _palette


class Processor:
    def __init__(self, path_to_weights):
        self.seg_tracker = self.__init_SegTracker(path_to_weights)


    def __save_prediction(self, pred_mask, output_dir, file_name):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask.save(os.path.join(output_dir,file_name))


    def __colorize_mask(self, pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)


    def __draw_mask(self, img, mask, alpha=0.5, id_countour=False):
        img_mask = np.zeros_like(img)
        img_mask = img
        if id_countour:
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids!=0]

            for id in obj_ids:
                color = _palette[id*3:id*3+3] if id <= 255 else [0,0,0]
                foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
                binary_mask = (mask == id)
                img_mask[binary_mask] = foreground[binary_mask]
                countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
                img_mask[countours, :] = 0
        else:
            binary_mask = (mask!=0)
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            foreground = img*(1-alpha)+self.__colorize_mask(mask)*alpha
            img_mask[binary_mask] = foreground[binary_mask]
            img_mask[countours,:] = 0    
        return img_mask.astype(img.dtype)


    def __create_dir(self, dir_path):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


    aot_model2ckpt = {
        "deaotb": "./ckpt/DeAOTB_PRE_YTB_DAV.pth",
        "deaotl": "./ckpt/DeAOTL_PRE_YTB_DAV",
        "r50_deaotl": "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
    }

    def __init_SegTracker(self, path_to_weights):
        aot_args["model"] = "r50_deaotl"
        #aot_args["model_path"] = "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth" #
        aot_args["model_path"] = path_to_weights #
        aot_args["long_term_mem_gap"] = 4
        aot_args["max_len_long_term"] = 9999
        segtracker_args["sam_gap"] = 9999
        segtracker_args['min_area'] = 250
        segtracker_args['min_new_obj_iou'] = 0.9
        segtracker_args["max_obj_num"] = 1
        sam_args["generator_args"]["points_per_side"] = 32
        
        seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
        seg_tracker.restart_tracker()
        return seg_tracker


    def __to_frame(self, img_bytes, grayscale=False):
        img_array = np.frombuffer(img_bytes.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if grayscale else frame
        return frame


    def __scale_frames(self, frame, scale):
        return [frame for i in range(scale)]


    def __init_first_frame(self, anchor_image, anchor_mask):
        with torch.cuda.amp.autocast():
            frame_idx = 0
            self.seg_tracker.restart_tracker()
            print(f"first_frame:: {anchor_image.shape}")
            print(f"first_mask:: {anchor_mask.shape}")
            self.seg_tracker.add_reference(anchor_image, anchor_mask, frame_idx)
            self.seg_tracker.first_frame_mask = anchor_mask


    def __get_frames_for_processing(self, anchor_image, anchor_mask, candidates, scale_screen):
        frames = []
        anchor_frame = self.__to_frame(anchor_image)
        anchor_mask = self.__to_frame(anchor_mask, grayscale=True)
        frames.append(anchor_frame)
        for screen_img in candidates:
            screen_frame = self.__to_frame(screen_img)
            screen_frames = self.__scale_frames(screen_frame, scale_screen)
            frames.extend(screen_frames)
        return frames, anchor_frame, anchor_mask


    def track_click(self, anchor_id, scroll_id, anchor_image, anchor_mask, candidates, fps, result_video_name, scale_screen=1):
        frames, anchor_frame, anchor_mask = self.__get_frames_for_processing(anchor_image, anchor_mask, candidates, scale_screen)
        tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{result_video_name}")}'
        self.__init_first_frame(anchor_frame, anchor_mask)
        self.__create_dir(tracking_result_dir)
        return self.__img_seq_type_input_tracking(anchor_id, scroll_id, frames, tracking_result_dir, result_video_name , fps)


    def __find_centroid(self, mask):
        mask = self.__colorize_mask(mask)
        red_pixels = np.where((mask[:, :, 0] != 0) & (mask[:, :, 1] != 0)& (mask[:, :, 2] != 0))
        if len(red_pixels[0]) < 50:
            return None, None 
        centroid_x = np.mean(red_pixels[1])
        centroid_y = np.mean(red_pixels[0])
        print(f'X::::::::::::::: {np.min(red_pixels[1])}, {np.max(red_pixels[1])}')
        print(f'Y::::::::::::::: {np.max(red_pixels[0])}, {np.min(red_pixels[0])}')
        return int(centroid_x), int(centroid_y)


    def __img_seq_type_input_tracking(self, anchor_id, scroll_id, frames, tracking_result_dir, result_video_name, fps):
        print(f':::::::::::PROCESSING:::::::::::::: {len(frames)}')
        x, y, frame_idx, frame_with_object = None, None, 0, 0
        pred_list = []
        sam_gap = self.seg_tracker.sam_gap
        output_mask_dir = f'{tracking_result_dir}/masks_{result_video_name}'
        output_masked_frame_dir = f'{tracking_result_dir}/frames_{result_video_name}'
        output_video = f'{tracking_result_dir}/vid_{result_video_name}.mp4'
        print(f'PUPA:::: {output_video}')
        self.__create_dir(output_mask_dir)
        self.__create_dir(output_masked_frame_dir)
        torch.cuda.empty_cache()
        gc.collect()
        with torch.cuda.amp.autocast():
            for frame in frames:
                frame_idx_real = frame_idx-1
                if frame_idx == 0:
                    pred_mask = self.seg_tracker.first_frame_mask
                    torch.cuda.empty_cache()
                    gc.collect()
                elif (frame_idx % sam_gap) == 0:
                    seg_mask = self.seg_tracker.seg(frame)
                    torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = self.seg_tracker.track(frame)
                    new_obj_mask = self.seg_tracker.find_new_objs(track_mask, seg_mask)
                    
                    self.__save_prediction(new_obj_mask, output_mask_dir, f'frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}_new.png')
                    print(f"processed %%% and saved frame frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}.png")
                    pred_mask = track_mask + new_obj_mask
                    self.seg_tracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = self.seg_tracker.track(frame, update_memory=True)
                torch.cuda.empty_cache()
                gc.collect()

                self.__save_prediction(pred_mask, output_mask_dir, f'frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}.png')
                pred_list.append(pred_mask)
                print(f"processed and saved frame frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}.png")

                if frame_idx != 0:
                    print(f"Find centroid for image::: {frame_idx_real}")
                    x, y = self.__find_centroid(pred_mask)
                    print(f"found coordinates::: {x}, {y}")
                if x!= None and y != None:
                    masked_frame = self.__draw_mask(frame, pred_mask)
                    cv2.imwrite(f"{output_masked_frame_dir}/frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}.png", masked_frame[:, :, ::-1])
                    print(f":::::COORS {x},{y}:::::{anchor_id}_scr:{scroll_id}_fr:{frame_idx_real}.png")
                    frame_with_object = frame_idx_real
                    break
                frame_idx += 1
            print('\nfinished')
        #height, width = pred_list[0].shape
        #fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
        #out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        #frame_idx = 0
        
        #for frame in frames:
        #    pred_mask = pred_list[frame_idx]
        #    masked_frame = self.__draw_mask(frame, pred_mask)
        #    masked_pred_list.append(masked_frame)
        #    cv2.imwrite(f"{output_masked_frame_dir}/frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx}.png", masked_frame[:, :, ::-1])
        #    masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        #    out.write(masked_frame)
        #    print(f'frame frame_anch:{anchor_id}_scr:{scroll_id}_fr:{frame_idx} writed')
        #    frame_idx += 1
        #    if frame_idx == frame_with_object:
        #        break
        #out.release()
        print("\n{} saved".format(output_video))
        print('\nfinished')
        torch.cuda.empty_cache()
        gc.collect()
        return frame_with_object, x, y