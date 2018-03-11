import random
import math
class RandomErase:
    '''
    Based on Random Erasing Data Augmention by Zhong et al.
    '''
    def __init__(self, area_ratio_range=[0.02, 0.4], aspect_ratio_factor=0.3, erase_prob=0.5  ):
        self.area_ratio_range=area_ratio_range
        self.aspect_ratio_factor = aspect_ratio_factor
        self.erase_prob= erase_prob
    def __call__(self, img_tensor):
        '''
        img tensor is expected to have dimensions 3 by width by height
        '''
        if random.uniform(0,1) > self.erase_prob:
            return img_tensor
        img_width = img_tensor.shape[1]
        img_height = img_tensor.shape[2]
        counter=0
        while True:
            area=img_width*img_height
            target_area = random.uniform(self.area_ratio_range[0], self.area_ratio_range[1])* area
            aspect_ratio = random.uniform(self.aspect_ratio_factor , 1/self.aspect_ratio_factor)
            width=math.ceil( math.sqrt(target_area*self.aspect_ratio_factor) )
            height=math.ceil( math.sqrt(target_area/self.aspect_ratio_factor  ) )
            if height>= img_height or width >= img_width:
                continue
            x = random.randint(0,img_width-width)
            y = random.randint(0,img_height-height)
            img_tensor[0,x:x+width, y:y+height]=random.uniform(0,1)
            img_tensor[1,x:x+width, y:y+height]=random.uniform(0,1)
            img_tensor[2,x:x+width, y:y+height]=random.uniform(0,1)
            return img_tensor
