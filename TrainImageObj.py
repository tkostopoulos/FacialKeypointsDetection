import numpy as np
import matplotlib.pyplot as plt


class TrainImage:
    def __init__(self, trainData, index):
        #################################################################################
        ############################### Defining Features ###############################
        #################################################################################
        self.leftEyeCenterFloat = (trainData["left_eye_center_y"][index],
                                   trainData["left_eye_center_x"][index])
        self.leftEyeCenterInt = (int(trainData["left_eye_center_y"][index]), 
                                 int(trainData["left_eye_center_x"][index]))
        
        self.leftEyeCantusFloat = (trainData["left_eye_inner_corner_y"][index], 
                                   trainData["left_eye_inner_corner_x"][index])
        self.leftEyeCantusInt = (int(trainData["left_eye_inner_corner_y"][index]), 
                                 int(trainData["left_eye_inner_corner_x"][index]))
        
        self.leftEyeOuterFloat = (trainData["left_eye_outer_corner_y"][index], 
                                  trainData["left_eye_outer_corner_x"][index])
        self.leftEyeOuterInt = (int(trainData["left_eye_outer_corner_y"][index]), 
                                 int(trainData["left_eye_outer_corner_x"][index]))
        
        self.rightEyeCenterFloat = (trainData["right_eye_center_y"][index], 
                                    trainData["right_eye_center_x"][index])
        self.rightEyeCenterInt = (int(trainData["right_eye_center_y"][index]), 
                                 int(trainData["right_eye_center_x"][index]))
        
        self.rightEyeCantusFloat = (trainData["right_eye_inner_corner_y"][index], 
                                   trainData["right_eye_inner_corner_x"][index])
        self.rightEyeCantusInt = (int(trainData["right_eye_inner_corner_y"][index]), 
                                 int(trainData["right_eye_inner_corner_x"][index]))
        
        self.rightEyeOuterFloat = (trainData["right_eye_outer_corner_y"][index], 
                                  trainData["right_eye_outer_corner_x"][index])
        self.rightEyeOuterInt = (int(trainData["right_eye_outer_corner_y"][index]),
                                 int(trainData["right_eye_outer_corner_x"][index])) 
        
        self.leftEyebrowInnerFloat= (trainData["left_eyebrow_inner_end_y"][index], 
                                     trainData["left_eyebrow_inner_end_x"][index])
        self.leftEyebrowInnerInt = (int(trainData["left_eyebrow_inner_end_y"][index]), 
                                 int(trainData["left_eyebrow_inner_end_x"][index]))
        
        self.leftEyebrowOuterFloat= (trainData["left_eyebrow_outer_end_y"][index], 
                                     trainData["left_eyebrow_outer_end_x"][index])
        self.leftEyebrowOuterInt = (int(trainData["left_eyebrow_outer_end_y"][index]), 
                                    int(trainData["left_eyebrow_outer_end_x"][index]))

        self.rightEyebrowInnerFloat= (trainData["right_eyebrow_inner_end_y"][index], 
                                      trainData["right_eyebrow_inner_end_x"][index])
        self.rightEyebrowInnerInt = (int(trainData["right_eyebrow_inner_end_y"][index]), 
                                     int(trainData["right_eyebrow_inner_end_x"][index]))
        
        self.rightEyebrowOuterFloat= (trainData["right_eyebrow_outer_end_y"][index], 
                                     trainData["right_eyebrow_outer_end_x"][index])
        self.rightEyebrowOuterInt = (int(trainData["right_eyebrow_outer_end_y"][index]), 
                                     int(trainData["right_eyebrow_outer_end_x"][index]))
        
        self.noiseTipFloat = (trainData["nose_tip_y"], trainData["nose_tip_x"][index])
        self.noiseTipInt = (int(trainData["nose_tip_y"][index]), 
                            int(trainData["nose_tip_x"][index]))
        
        self.mouthLeftFloat = (trainData["mouth_left_corner_y"][index], 
                               trainData["mouth_left_corner_x"][index])
        self.mouthLeftInt = (int(trainData["mouth_left_corner_y"][index]), 
                             int(trainData["mouth_left_corner_x"][index]))
        
        self.mouthRightFloat = (trainData["mouth_right_corner_y"][index], 
                                trainData["mouth_right_corner_x"][index])
        self.mouthRightInt = (int(trainData["mouth_right_corner_y"][index]), 
                             int(trainData["mouth_right_corner_x"][index]))
        
        self.mouthTopFloat = (trainData["mouth_center_top_lip_y"][index], 
                              trainData["mouth_center_top_lip_x"][index])
        self.mouthTopInt = (int(trainData["mouth_center_top_lip_y"][index]), 
                            int(trainData["mouth_center_top_lip_x"][index]))
                
        self.mouthBottomFloat = (trainData["mouth_center_bottom_lip_y"][index], 
                                 trainData["mouth_center_bottom_lip_x"][index])
        self.mouthBottomInt = (int(trainData["mouth_center_bottom_lip_y"][index]), 
                               int(trainData["mouth_center_bottom_lip_x"][index]))
        
        self.image = self.formatImage(trainData["Image"][index])
        self.labeledImage = self.LabeledImage()
        
    def formatImage(self, ImageStrArray, imageSize = (96,96)):
        '''
        Purpose: take the long string, separate it by the spaces and turn it into an image
        '''
        imageParse = ImageStrArray.split(' ')
        imageInt = np.array(imageParse, dtype=int)
        image = np.reshape(imageInt, imageSize) # defined in the database
        return image
    
    def ShowImage(self, cmap = 'Spectral'):
        '''
	    Purpose: show the image
        '''
        plt.figure()
        plt.imshow(self.image)
        plt.show()
        
        plt.figure()
        plt.imshow(self.labeledImage, cmap=cmap)
        plt.show()
    
    def LabeledImage(self, threshold = 50):
        '''
        Purpose: apply the labels to the image
        Improvement: diolate the dots to make them easier to see
        '''
        temp_image = np.array(self.image) #MUST REFORMAT SO THAT IT DOESNT USE SAME DATA
        temp_image[temp_image < threshold ] = threshold
        # if less that the threshold reset the threshold
        # to make this easier to see on the labeled image

        temp_image[self.leftEyebrowInnerInt] = 0 
        temp_image[self.leftEyebrowOuterInt] = 0
        
        temp_image[self.leftEyeCantusInt] = 0 
        temp_image[self.leftEyeCenterInt] = 0
        temp_image[self.leftEyeOuterInt] = 0 
        
        temp_image[self.mouthBottomInt] = 0
        temp_image[self.mouthLeftInt] = 0
        temp_image[self.mouthRightInt] = 0
        temp_image[self.mouthTopInt] = 0
        
        temp_image[self.rightEyebrowInnerInt] = 0
        temp_image[self.rightEyebrowOuterInt] = 0
        
        temp_image[self.rightEyeCantusInt] = 0 
        temp_image[self.rightEyeCenterInt] = 0
        temp_image[self.rightEyeOuterInt] = 0
        
        temp_image[self.noiseTipInt] = 0
        
        return temp_image