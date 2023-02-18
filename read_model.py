import segmentation_models_pytorch as smp

def define_model(model_name,encoder_name='resnet34',encoder_weights='imagenet',in_channels=1,classes=8):
  if model_name == 'unet':
    model = smp.Unet(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=8,                      # model output channels (number of classes in your dataset)
    )
  elif model_name=='unetplusplus':
      
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=8,                      # model output channels (number of classes in your dataset)
    )
  
    return model  