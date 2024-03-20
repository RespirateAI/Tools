import os
import cv2

image_folder = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/ALL'  # Replace 'your_image_folder_path' with the path to your image folder
txt_file = 'train_dataset.txt'  # Replace 'your_text_file.txt' with the path to your text file
first_T_dir = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Training/Attention_Maps/Testing_Pipeline/Resnet/1st Tier'
both_T_dir = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/Both Tiers'
save_heatmaps_dir = '/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/overlapped_attention'
opacity = 0.55

def crop_heatmap_resize(heatmap,image):
    image_height,image_width,alpha = image.shape
    heatmap_height, heatmap_width,alpha = heatmap.shape
    crop_top = 96   # Number of pixels to crop from the top
    crop_bottom = 88  # Number of pixels to crop from the bottom
    crop_left = 130  # Number of pixels to crop from the left side
    crop_right = 255  # Number of pixels to crop from the right side
    heatmap_cropped = heatmap[crop_top:heatmap_height - crop_bottom, crop_left:heatmap_width - crop_right]
    heatmap_resized_cropped = cv2.resize(heatmap_cropped, (image.shape[1], image.shape[0]))
    return heatmap_resized_cropped

for file in os.listdir('/media/wenuka/New Volume-G/01.FYP/Tool_1/Test_Pipeline/dataset/splitted/test'):
    image_path = os.path.join(image_folder,file[:-4])
    image = cv2.imread(image_path)
    print(f'processing {file}')
    both_T_all = os.path.join(both_T_dir,'All',file[:-4]+".png")
    both_T_all_heatmap = cv2.imread(both_T_all)
    both_T_all_heatmap = crop_heatmap_resize(both_T_all_heatmap,image)
    blended_both_T_all = cv2.addWeighted(image, 1 - opacity, both_T_all_heatmap, opacity, 0)

    cv2.imwrite(os.path.join(save_heatmaps_dir,'Resnet','Both Tiers','All',file[:-4]+".png"),blended_both_T_all)


# with open(txt_file, 'r') as file:
#     for line in file:
#         if len(line.strip())>0:
#             filename, value_str = line.strip().split('=')
#             image_path = os.path.join(image_folder, filename+".png")
#             if os.path.exists(image_path):
#                 value = int(value_str)
#                 image = cv2.imread(image_path)
#                 slice_name = f'{filename}_slice{value}.png'
                
#                 first_T_all = os.path.join(first_T_dir,'All',slice_name)
#                 first_T_all_heatmap = cv2.imread(first_T_all)
#                 first_T_all_heatmap = crop_heatmap_resize(first_T_all_heatmap,image)
#                 blended_first_T_all = cv2.addWeighted(image, 1 - opacity, first_T_all_heatmap, opacity, 0)

#                 first_T_avg = os.path.join(first_T_dir,'AVG',filename+".png")
#                 first_T_avg_heatmap = cv2.imread(first_T_avg)
#                 first_T_avg_heatmap = crop_heatmap_resize(first_T_avg_heatmap,image)
#                 blended_first_T_avg = cv2.addWeighted(image, 1 - opacity, first_T_avg_heatmap, opacity, 0)

#                 both_T_all = os.path.join(both_T_dir,'All',slice_name)
#                 both_T_all_heatmap = cv2.imread(both_T_all)
#                 both_T_all_heatmap = crop_heatmap_resize(both_T_all_heatmap,image)
#                 blended_both_T_all = cv2.addWeighted(image, 1 - opacity, both_T_all_heatmap, opacity, 0)


#                 both_T_avg = os.path.join(both_T_dir,"AVG",filename+".png")
#                 both_T_avg_heatmap = cv2.imread(both_T_avg)
#                 both_T_avg_heatmap = crop_heatmap_resize(both_T_avg_heatmap,image)
#                 blended_both_T_avg = cv2.addWeighted(image, 1 - opacity, both_T_avg_heatmap, opacity, 0)

#                 print(os.path.join(save_heatmaps_dir,'Resnet','1st Tier','All'))
#                 cv2.imwrite(os.path.join(save_heatmaps_dir,'Resnet','1st Tier','All',filename[:-4]+".png"),blended_first_T_all)
#                 cv2.imwrite(os.path.join(save_heatmaps_dir,'Resnet','1st Tier','AVG',filename[:-4]+".png"),blended_first_T_avg)
#                 cv2.imwrite(os.path.join(save_heatmaps_dir,'Resnet','Both Tiers','All',filename[:-4]+".png"),blended_both_T_all)
#                 cv2.imwrite(os.path.join(save_heatmaps_dir,'Resnet','Both Tiers','AVG',filename[:-4]+".png"),blended_both_T_avg)
#             else:
#                 print(f'{image_path} does not exist')




            