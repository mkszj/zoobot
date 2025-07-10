import torch
import timm
from timm.optim import create_optimizer_v2, param_groups_layer_decay
from timm.scheduler import create_scheduler_v2

if __name__ == "__main__":

    # encoder = timm.create_encoder('vit_small_patch16_224', pretrained=True)
    encoder = timm.create_model('convnext_nano.in12k', pretrained=True)
    # print(encoder)  # Print the encoder architecture

    head = torch.nn.Linear(encoder.num_features, 1000)  # Example head for classification

    # param_groups = param_groups_layer_decay(encoder)
    # print(param_groups) 
    # for group in param_groups:
    #     print(group['lr_scale'], group['weight_decay'], [p.size() for p in param_groups[3]['params']])

    optimizer = create_optimizer_v2(encoder, 'adamw', lr=0.001, weight_decay=0.01)
    print(len(optimizer.param_groups))

    # add head parameters to optimizer
    optimizer.add_param_group({'params': head.parameters(), 'lr': 0.001})
    print(len(optimizer.param_groups))

    print(optimizer)  # Print the optimizer configuration

    scheduler = create_scheduler_v2(optimizer, 'cosine', warmup_epochs=5)
    print(scheduler)  # Print the scheduler configuration
