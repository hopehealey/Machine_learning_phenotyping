import importlib
import torch
import torch.nn.functional as F

from torchvision import datasets
import torchvision
import torchvision.models as models

from dac.utils import image_to_tensor

def init_network(checkpoint_path=None, input_shape=(224,224), net_module="ResNet", 
                 input_nc=3, output_classes=5, gpu_ids=[], eval_net=True, require_grad=False,
                 downsample_factors=None):
    """
    checkpoint_path: Path to train checkpoint to restore weights from
    input_nc: input_channels for aux net
    aux_net: name of aux net
    """
    #print("THE CHECK", checkpoint_path)

    net_mod = importlib.import_module(f"dac_networks.{net_module}")
    net_class = getattr(net_mod, f'{net_module}')
    if net_module == "Vgg2D":
        net = net_class(input_size=input_shape, input_channels=input_nc, output_classes=output_classes,
                        downsample_factors=downsample_factors)
    else:
        net = net_class(input_size=input_shape, input_channels=input_nc, output_classes=output_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    net = torchvision.models.resnet18(pretrained = False, num_classes=5)

    if eval_net:
        net.eval()

    if require_grad:
        for param in net.parameters():
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = False

    if checkpoint_path is not None:
        #print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        #print(checkpoint)
        try:
            net.load_state_dict(checkpoint)
        except KeyError:
            net.load_state_dict(checkpoint)
    return net


def run_inference(net, im):
    """
    Net: network object
    input_image: Normalized 2D input image.
    """
    im_tensor = image_to_tensor(im)
    class_probs = F.softmax(net(im_tensor), dim=1)
    return class_probs
