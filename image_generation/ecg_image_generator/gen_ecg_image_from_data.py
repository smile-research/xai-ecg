import os, sys, json
import random
import qrcode
from PIL import Image
import numpy as np
from ecg_image_generator.HandwrittenText.generate import get_handwritten
from ecg_image_generator.CreasesWrinkles.creases import get_creased
from ecg_image_generator.ImageAugmentation.augment import get_augment
import warnings
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

def run_single_file(
    input_file: str,
    output_file: str,
    add_qr_code: bool = False,
    qr_code_seed: Optional[int] = None,
    link: str = "",
    num_words: int = 5,
    x_offset: int = 30,
    y_offset: int = 30,
    handwriting_size_factor: float = 0.2,
    crease_angle: int = 90,
    num_creases_vertically: int = 10,
    num_creases_horizontally: int = 10,
    rotate: int = 0,
    noise: int = 50,
    crop: float = 0.01,
    lead_name_bbox: bool = False,
    store_config: int = 0,
    deterministic_offset: bool = False,
    deterministic_num_words: bool = False,
    deterministic_hw_size: bool = False,
    deterministic_angle: bool = False,
    deterministic_vertical: bool = False,
    deterministic_horizontal: bool = False,
    deterministic_rot: bool = False,
    deterministic_noise: bool = False,
    deterministic_crop: bool = False,
    deterministic_temp: bool = False,
    fully_random: bool = False,
    hw_text: bool = False,
    wrinkles: bool = False,
    augment: bool = False,
    lead_bbox: bool = False
) -> int:
    """
    Processes a single file by applying various transformations such as handwritten text,
    creases/wrinkles, augmentation, and QR code addition based on the provided parameters.
    
    Parameters:
        input_file (str): Path to the input file.
        add_qr_code (bool, optional): Whether to add a QR code. Defaults to False.
        qr_code_seed (Optional[int], optional): Seed for QR code generation. Defaults to None.
        link (str, optional): Link used for handwritten text generation. Defaults to "".
        num_words (int, optional): Number of words for handwritten text. Defaults to 5.
        x_offset (int, optional): X-axis offset for text placement. Defaults to 30.
        y_offset (int, optional): Y-axis offset for text placement. Defaults to 30.
        handwriting_size_factor (float, optional): Scale factor for handwritten text. Defaults to 0.2.
        crease_angle (int, optional): Angle of creases. Defaults to 90.
        num_creases_vertically (int, optional): Number of vertical creases. Defaults to 10.
        num_creases_horizontally (int, optional): Number of horizontal creases. Defaults to 10.
        rotate (int, optional): Rotation angle. Defaults to 0.
        noise (int, optional): Noise level. Defaults to 50.
        crop (float, optional): Crop factor. Defaults to 0.01.
        lead_name_bbox (bool, optional): Whether to add bounding boxes to lead names. Defaults to False.
        store_config (int, optional): Per-file config saving. Saves what transformations were performed on a file. If this function is run multiple files for a file, the new config overwrites the old config. The config is stored at the same folder as the modified image. 0 - doesn't store config. 1 - saves empty config. 2 - saves the transformations' details to config. Defaults to 0.
        deterministic_offset (bool, optional): Use deterministic offset. Defaults to False.
        deterministic_num_words (bool, optional): Use deterministic word count. Defaults to False.
        deterministic_angle (bool, optional): Use deterministic crease angle. Defaults to False.
        deterministic_vertical (bool, optional): Use deterministic vertical creases. Defaults to False.
        deterministic_horizontal (bool, optional): Use deterministic horizontal creases. Defaults to False.
        deterministic_noise (bool, optional): Use deterministic noise. Defaults to False.
        fully_random (bool, optional): Apply fully random transformations. Defaults to False.
        hw_text (bool, optional): Add handwritten text. Defaults to False.
        wrinkles (bool, optional): Apply wrinkles effect. Defaults to False.
        augment (bool, optional): Apply image augmentation. Defaults to False.
        lead_bbox (bool, optional): Apply bounding box to leads. Defaults to False.
        deterministic_temp (bool) - unused.
        deterministic_crop (bool) - unused.
        deterministic_rot (bool) - unused.
        deterministic_hw_size (bool) - unused.        

    Returns:
        int: Number of processed files.
    """
    output_directory = '.' # argument used below, doesn't do anything, so I removed it from the function parameter list
    
    # Script messes up with relative paths if execution directory is different than expected.
    # We set it to the script's parent folder here and change it back at the end of the script so it doesn't mess up your script, hopefully.
    old_cwd = os.getcwd()
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    
    if store_config:
        rec_tail, extn = os.path.splitext(input_file)
        try:
            with open(rec_tail + '.json', 'r') as file:
                json_dict = json.load(file)
        except FileNotFoundError:
            json_dict = {}  # Default to None if the file is missing
    else:
        json_dict = {}
    if(fully_random):
        hw_text = random.choice((True,False))
        wrinkles = random.choice((True,False))
        augment = random.choice((True,False))
    else:
        hw_text = hw_text
        wrinkles = wrinkles
        augment = augment
    
    #Handwritten text addition
    if(hw_text):
        num_words = num_words if (deterministic_num_words) else random.choice(range(2,num_words+1))
        x_offset = x_offset if (deterministic_offset) else random.choice(range(1,x_offset+1))
        y_offset = y_offset if (deterministic_offset) else random.choice(range(1,y_offset+1))

        input_file = get_handwritten(link=link,num_words=num_words,input_file=input_file,output_dir=output_directory,x_offset=x_offset,y_offset=y_offset,handwriting_size_factor=handwriting_size_factor,bbox = lead_bbox, output_path=output_file)
        output_file = None
    else:
        num_words = 0
        x_offset = 0
        y_offset = 0

    if store_config == 2:
        json_dict['handwritten_text'] = bool(hw_text)
        json_dict['num_words'] = num_words
        json_dict['x_offset_for_handwritten_text'] = x_offset
        json_dict['y_offset_for_handwritten_text'] = y_offset
    
    if(wrinkles):
        ifWrinkles = True
        ifCreases = True
        crease_angle = crease_angle if (deterministic_angle) else random.choice(range(0,crease_angle+1))
        num_creases_vertically = num_creases_vertically if (deterministic_vertical) else random.choice(range(1,num_creases_vertically+1))
        num_creases_horizontally = num_creases_horizontally if (deterministic_horizontal) else random.choice(range(1,num_creases_horizontally+1))
        input_file = get_creased(input_file,output_directory=output_directory,ifWrinkles=ifWrinkles,ifCreases=ifCreases,crease_angle=crease_angle,num_creases_vertically=num_creases_vertically,num_creases_horizontally=num_creases_horizontally,bbox = lead_bbox, output_path=output_file)
    else:
        crease_angle = 0
        num_creases_horizontally = 0
        num_creases_vertically = 0

    if store_config == 2:
        json_dict['wrinkles'] = bool(wrinkles)
        json_dict['crease_angle'] = crease_angle
        json_dict['number_of_creases_horizontally'] = num_creases_horizontally
        json_dict['number_of_creases_vertically'] = num_creases_vertically

    if(augment):
        noise = noise if (deterministic_noise) else random.choice(range(1,noise+1))
    
        if(not lead_bbox):
            do_crop = random.choice((True,False))
            if(do_crop):
                crop = crop
            else:
                crop = crop
        else:
            crop = 0
        blue_temp = random.choice((True,False))

        if(blue_temp):
            temp = random.choice(range(2000,4000))
        else:
            temp = random.choice(range(10000,20000))
        rotate = rotate
        input_file = get_augment(input_file,output_directory=output_directory,rotate=rotate,noise=noise,crop=crop,temperature=temp,bbox = lead_bbox, store_text_bounding_box = lead_name_bbox, json_dict = json_dict)
    
    else:
        crop = 0
        temp = 0
        rotate = 0
        noise = 0
    if store_config == 2:
        json_dict['augment'] = bool(augment)
        json_dict['crop'] = crop
        json_dict['temperature'] = temp
        json_dict['rotate'] = rotate
        json_dict['noise'] = noise

    if store_config:
        json_object = json.dumps(json_dict, indent=4)
        
        with open(rec_tail + '.json', "w") as f:
            f.write(json_object)


    if add_qr_code:
        img = np.array(Image.open(input_file))
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=5,
            border=4,
        )
        if qr_code_seed:
            random.seed(qr_code_seed)
        encoding = input_file
        qr.add_data(encoding)
        qr.make(fit=True)

        qr_img = np.array(qr.make_image(fill_color="black", back_color="white"))
        qr_img_color = np.zeros((qr_img.shape[0], qr_img.shape[1], 3))
        qr_img_color[:,:,0] = qr_img*255.
        qr_img_color[:,:,1] = qr_img*255.
        qr_img_color[:,:,2] = qr_img*255.
        
        img[:qr_img.shape[0], -qr_img.shape[1]:, :3] = qr_img_color
        img = Image.fromarray(img)
        img.save(output_file)
    
    os.chdir(old_cwd) # restore old execution directory.
    return 1

if __name__=='__main__':
    # # run wrinkles only:
    run_single_file(
        input_file='/workspaces/export-package/img_to_modify/file_to_modify.png',
        # wrinkles=True,
        # crease_angle=45,
        # add_qr_code=True,
        # store_config=2,
        hw_text=True,
    )


    
