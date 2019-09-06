import torch
import torch.utils
import torch.nn as nn
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TYolo(nn.Module):
    def __init__(self):
        super(TYolo,self).__init__()
        
        self.conv1=nn.Conv2d(3,16,3,padding=0)
        self.conv2=nn.Conv2d(16,32,3,padding=0)
        self.conv3=nn.Conv2d(32,64,3,padding=0)
        self.conv4=nn.Conv2d(64,128,3,padding=0)
        self.conv5=nn.Conv2d(128,256,3,padding=0)
        self.conv6=nn.Conv2d(256,512,3,padding=0)
        self.conv7=nn.Conv2d(512,1024,3,padding=0)
        self.conv8=nn.Conv2d(1024,32,3,padding=0)
        self.conv9=nn.Conv2d(32,30,1,padding=0)
        self.pool1=nn.MaxPool2d(2,2)
        self.pool2=nn.MaxPool2d((2,2),(1,1))

    def forward(self, x):
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv1(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv2(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv3(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv4(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=self.pool1(F.leaky_relu(self.conv5(x)))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv6(x))
        x=F.pad(x,(0,1,0,1),'reflect')
        x=self.pool2(x)
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv7(x))
        x=F.pad(x,(1,1,1,1),'reflect')
        x=F.leaky_relu(self.conv8(x))
        x=self.conv9(x)
        return x
        
def predict_transform(pred, CUDA = True):
    anchors = [[1.08,1.19],  [3.42,4.41],  [6.63,11.38],  [9.42,5.11],  [16.62,10.52]]
    batch_size = pred.size(0)
    stride =  416 // pred.size(2)
    grid_size = 416//stride
    bbox_attrs = 5 + 1
    num_anchors = len(anchors)
    
    prediction = pred.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    anchors = torch.FloatTensor(anchors)
    prediction[:,:,:2] += x_y_offset
    
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 6] = torch.sigmoid((prediction[:,:, 5 : 6]))
    
    prediction[:,:,:4] *= stride
    prediction[:,:,:4] /= 416
    
    return prediction

def bb_intersection_over_union(pred, gt):
    boxA = [pred[0]-pred[2]/2, pred[1]-pred[3]/2, pred[0]+pred[2]/2, pred[1]+pred[3]/2]
    boxB = [gt[0]-gt[2]/2, gt[1]-gt[3]/2, gt[0]+gt[2]/2, gt[1]+gt[3]/2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou    
  
#curently desn't work  
def yolo_loss(pred,labels):
    
    pred=predict_transform(pred)[0]
    labels=labels[0]
    print(labels.shape,pred.shape)
    col=ceil(labels[0]*13)
    row=ceil(labels[1]*13)
    rng=(col*row-1)*5
    def sumsqr(out,target):
        return torch.sum((out-target)**2)
    mse=sumsqr
    print(labels)
    #col=3
    #row=1

    #iou
#     true_xy=labels[:2]
#     true_wh=labels[2:4]
    
#     true_mins=true_xy-true_wh/2
#     true_max=true_xy+true_wh/2
    
#     pred_xy=pred[:,:2]
#     pred_wh=pred[:,2:4]
    
#     pred_mins=pred_xy-pred_wh/2
#     pred_max=pred_xy+pred_wh/2

#     intersect_mins=torch.max(pred_mins,true_mins)
#     intersect_max=torch.min(pred_max,true_max)
#     intersect_wh=torch.max(intersect_max-intersect_mins,0)[0]
#     intersect_areas=intersect_wh[0]*intersect_wh[1]
#     print(intersect_areas)
#     true_area=true_wh[0]*true_wh[1]
#     pred_area=pred_wh[:,0]*pred_wh[:,1]
#     union_areas=pred_area+true_area-intersect_areas
#     iou_scores=torch.div(intersect_areas,union_areas)
    print(pred)
    iou_scores=[]
    for x in range(len(pred)):
        iou_scores.append(bb_intersection_over_union(pred[x],labels))
    iou_scores=torch.tensor(iou_scores)
    print(iou_scores)
    iou_scores=torch.Tensor(iou_scores).to(device)
    
    loss_x=5*torch.sum((pred[rng:rng+5,0]-labels[0])**2)
    loss_y=5*torch.sum((pred[rng:rng+5,1]-labels[1])**2)
    loss_w=torch.sum((pred[rng:rng+5,2]-labels[2])**2)
    loss_h=torch.sum((pred[rng:rng+5,3]-labels[3])**2)
    loss_conf=0.5*(torch.sum(pred[:rng,4]-iou_scores[:rng])**2)+torch.sum((pred[rng:rng+5,4]-iou_scores[rng:rng+5])**2)+torch.sum((pred[rng+5:,4]-iou_scores[rng+5:])**2)    
    print("loss x,y,w,h,",loss_x,loss_y,loss_w,loss_h)
    loss=torch.sum(loss_x,loss_y)+torch.sum(loss_w,loss_h)
    
#     for i in range(1,rng):
#         loss+=((pred[i,4]-iou_scores[i])**2)
#     for i in  range(rng,rng+5):
#         loss+=((pred[i,0]-labels[0])**2
#                     +(pred[i,1]-labels[1])**2)
#     for i in range(rng,rng+5):
#         loss+=((torch.sqrt(pred[i,2])-torch.sqrt(labels[2]))**2+(torch.sqrt(pred[i,3])-torch.sqrt(labels[3]))**2)
#     for i in range(rng,rng+5):
#         loss+=((pred[i,4]-iou_scores[i])**2)
#     for i in range(rng+5,845):
#         loss+=((pred[i,4]-iou_scores[i])**2)
    return loss