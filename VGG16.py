import keras as K
import os 

class VGG16_Obj:
    def __init__ (self, projectDirectory = "C:\\Users\\Ted\\Projects\\Python\\FacialKeypointsDetection"):
        self.projectDirectory = projectDirectory
        self.modelFC   = self.VGG16()
        self.modelNoFC = self.VGG16(FC_Include = False) # add additional method to call this out
        self.modelReducedFC = self.AddFCtoVGG16FeatureExtractor()

    def VGG16(self,
              ClassicVGG16=True, 
              FC_Include = True,
              l2_weight = 5e-04,
              like_its_hot = 0.7, # drop regulation
              FeatureExtractorTraining = False, 
              FCTraining = True,
              weights= 'imagenet', 
              input_tensor=None): 
        ''' 
        # Inputs
        FC_Include = using the network as a feature extractor based on the convolutional layers
        FullyConnected = if training is needed for the fully connected layer
        classificationNumber = number of components 
        FeatureExtractorTraining = if you need to train the middle layers
        weights = 'imagenet' means using the weights from a network pretrained by imagenet challenge
        
        Rules:
        CANNOT have MORE than 1000 outputs (can't really see a case where they're would be >1k but eat your heart out.
        Use the excess and keep training on the FC layers true
        
        # Returns
            VGG16 Network
        '''
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')
        # Determine proper input shape
        if ClassicVGG16:
            inputShape = (224, 224, 3)
        else:
            inputShape = (None, None, 3)
        
        img_input = K.Input(inputShape)
        
        # Block 1     
        b1_1 = K.layers.Conv2D(64, (3, 3), 
                      activation='relu', 
                      padding='same', # border_mode is now padding
                      name='block1_conv1')
        b1_1.trainable = FeatureExtractorTraining
        x = b1_1(img_input)
        
        b1_2 = K.layers.Conv2D(64, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block1_conv2')
        b1_2.trainable = FeatureExtractorTraining
        x = b1_2(x)#_normalized)
        
        x = K.layers.MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x) 
        
        # Block 2
        b2_1 = K.layers.Conv2D(128, 
                      (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block2_conv1')
        b2_1.trainable = FeatureExtractorTraining
        x = b2_1(x)
        
        b2_2 = K.layers.Conv2D(128, (3, 3), 
                               activation='relu', 
                               padding='same', 
                               name='block2_conv2')
        b2_2.trainable = FeatureExtractorTraining
        x = b2_2(x)#_normalized)
        
        x = K.layers.MaxPooling2D((2,2), strides=(2,2) , name='block2_pool')(x)#_normalized) # decrease the amout of data points with no rounding loss
        
        # Block 3
        # convolution block
        
        b3_1 = K.layers.Conv2D(256, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block3_conv1')
        b3_1.trainable = FeatureExtractorTraining
        x = b3_1(x)
        
        b3_2 = K.layers.Conv2D(256, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block3_conv2')
        b3_2.trainable = FeatureExtractorTraining
        x = b3_2(x)#_normalized)

        b3_3 = K.layers.Conv2D(256, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block3_conv3')
        b3_3.trainable = FeatureExtractorTraining
        x = b3_3(x)#_normalized)
        
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4 identity doc
        
        b4_1 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block4_conv1')
        b4_1.trainable = FeatureExtractorTraining
        x = b4_1(x)
        
        b4_2 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block4_conv2')
        b4_2.trainable = FeatureExtractorTraining
        x = b4_2(x)#_normalized)

        b4_3 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block4_conv3')
        b4_3.trainable = FeatureExtractorTraining
        x = b4_3(x)#_normalized)
        
        x = K.layers.MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
        
        #Block 5
        b5_1 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block5_conv1')
        b5_1.trainable = FeatureExtractorTraining
        x = b5_1(x)
        
        b5_2 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block5_conv2')
        b5_2.trainable = FeatureExtractorTraining
        x = b5_2(x)
        
        b5_3 = K.layers.Conv2D(512, (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='block5_conv3')
        b5_3.trainable = FeatureExtractorTraining
        x = b5_3(x)
        
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        x = K.layers.Flatten(name='flatten')(x)

        if FC_Include:
            # Classification block
            #x = K.layers.Flatten(name='flatten')(x) # moved this inside the lines
            
            #x = Dropout(like_its_hot, name = 'regulator_0')(x)
            fc1 = K.layers.Dense(4096, activation='relu',
                        kernel_regularizer=K.regularizers.l2(l2_weight),
                        name='fc1')
            fc1.trainable = FCTraining
            x = fc1(x)
            
            x = K.layers.Dropout(like_its_hot, name = 'regulator_1')(x)
            
            fc2 = K.layers.Dense(4096, 
                        activation='relu', 
                        kernel_regularizer=K.regularizers.l2(l2_weight),
                        name='fc2')
            fc2.trainable = FCTraining
            x = fc2(x)
            
            x = K.layers.Dropout(like_its_hot, name = 'regulator_2')(x)
            
            pred  = K.layers.Dense(1000, 
                          activation='softmax', 
                          kernel_regularizer=K.regularizers.l2(l2_weight),
                          name='pred')(x)
            
            model = K.Model(img_input, pred)
            
        else: ########################################################################################
            #print ("You got no legs Lieutenant Dan!!!")
            model = K.Model(img_input,x)
            
        # load weights
        if weights == 'imagenet':
            currentCwd = os.getcwd()
            os.chdir(self.projectDirectory) # hard coded for my directory
            if FC_Include == False:
                modelWeights = model.load_weights('vgg16Weights_noFC.h5')            
            elif FC_Include == True: # only include the top if has 
                modelWeights = model.load_weights('vgg16Weights_FCincluded.h5')
            os.chdir(currentCwd)
        return model


    def AddFCtoVGG16FeatureExtractor(self, fc1Variable = 200, 
    	                            fc2Variable = 80, predVariable = 30, 
    	                            l2_weight = 5e-04, like_its_hot = 0.1):
        '''
        Purpose: utalize Transfer Learning and slam a vgg16 network together with different FC layers
        Output: model with just that (if you do a model.summary it misses the new model but data is there)
        '''
        model1 = self.modelNoFC # get the pretrained network with no FC layer
        # the majic of transfer learning
        
        #make the second model we slam together
        model2 = K.models.Sequential() # MUST use Keras API to add layers together
        model2.add(K.layers.Dense(fc1Variable, #define amount of variables in function header
                      activation='relu',
                      kernel_regularizer=K.regularizers.l2(l2_weight),
                      name  ="fc1"))
        model2.add(K.layers.Dropout(like_its_hot, name = 'regulator_1')) # add regulation
        model2.add(K.layers.Dense(fc2Variable, #define amount of variables in function header
                     activation='relu',
                     kernel_regularizer=K.regularizers.l2(l2_weight),
                     name ="fc2"))
        model2.add(K.layers.Dropout(like_its_hot, name = 'regulator_2')) #add a little more regulation
        model2.add(K.layers.Dense(predVariable, #define amount of variables in function header
                      #activation='softmax', # don't want an activator function here
                      kernel_regularizer=K.regularizers.l2(l2_weight),
                      name  ="pred"))

        linkingOutput = model2(model1.output) #link the two models here
        finalModel = K.Model(model1.input, linkingOutput)
        return finalModel
        