import keras as K

class VGG16_Obj:
	def __init__ (self):
		self.modelFC = VGG16()
        self.modelNoFC = VGG16() # add additional method to call this out
		

	def VGG16(ClassicVGG16=True, 
	          FC_Include = True,
	          classificationNumber = 10,
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
	                  border_mode='same', 
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
	                  border_mode='same', 
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

	    if FC_Include:
	        # Classification block
	        x = K.layers.Flatten(name='flatten')(x)
	        
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
	        
	        pred  = K.layers.Dense(classificationNumber, 
	                      activation='softmax', 
	                      kernel_regularizer=K.regularizers.l2(l2_weight),
	                      name='pred')(x)
	        
	        model = K.Model(img_input, pred)
	        
	    else: ########################################################################################
	        print ("You got no legs Lieutenant Dan!!!")
	        model = K.Model(img_input,x)
	        
	    # load weights
	    if weights == 'imagenet':
	        currentCwd = os.getcwd()
	        os.chdir(projectDirectory) # hard coded for my directory
	        if FC_Include == False:
	            modelWeights = model.load_weights('vgg16Weights_noFC.h5')            
	        elif FC_Include == True: # only include the top if has 
	            modelWeights = model.load_weights('vgg16Weights_FCincluded.h5')
	        os.chdir(currentCwd)
	    return model