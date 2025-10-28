# ex1MNIST.py
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def inference(weight,input):
    output = torch.matmul(input, weight)
    output = torch.sigmoid(output)
    return output

class softmaxwithloss():
    def __init__(self,input,target):
        self.input = input
        self.target = target
        self.softed=None
        self.loss_value = 0.0


    def forward(self):
        self.softed = torch.softmax(self.input, dim=1)
        return self.softed
    
    def backward(self):
        self.grad_input = (self.softed - self.target)/ self.target.size(0)
        return self.grad_input

    def loss(self):
        self.loss_value = -torch.sum(self.target * torch.log(self.softed + 1e-10)) / self.target.size(0)
        #print(f"loss: {self.loss_value}")

class affine():
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.input =None

    def forward(self, input):
        self.input =input
        return torch.matmul(input, self.weight) + self.bias

    def backward(self, grad_output):
        grad_input = torch.matmul(grad_output, self.weight.t())
        grad_weight = torch.matmul(self.input.t(), grad_output)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias
    #偏置的维度和权重是不同的，该怎么处理？ 广播机制
class sigmoid():
    def __init__(self):
        self.inout = None
        self.output = None

    def forward(self,input):
        self.inout = input
        self.output=torch.sigmoid(input)
        return self.output

    def backward(self,dout):
        # 正确的 sigmoid 反向传播梯度计算
        grad = dout * self.output * (1 - self.output)
        return grad

class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        return torch.relu(input)

    def backward(self,grad_output):
        grad_input = grad_output.clone()
        grad_input= grad_input * (grad_output > 0).float()
        return grad_input


class twowisenetwork():
    def __init__(self, input_size, hidden_size, output_size,stdrate=0.01):
        self.params={}
        self.params['weight1'] = torch.randn(input_size, hidden_size) * torch.sqrt(torch.tensor(2. / input_size))
        self.params['weight2'] = torch.randn(hidden_size, output_size) * torch.sqrt(torch.tensor(2. / hidden_size))
        self.params['bias1'] = torch.zeros(hidden_size)+0.1
        self.params['bias2'] = torch.zeros(output_size)+0.1
        self.params['std'] = stdrate
        self.layers={}
        self.layers['affine1']=affine(self.params['weight1'], self.params['bias1'])
        self.layers['ReLU']=ReLU()
        self.layers['affine2']=affine(self.params['weight2'], self.params['bias2'])
        
        self.grad = {}
        self.inference=None
        self.inferenceout=None
    def forward(self, x):
        x=x.squeeze()
        x=self.layers['affine1'].forward(x)
        x=self.layers['ReLU'].forward(x)
        x=self.layers['affine2'].forward(x)
        self.inference=x
        self.inferenceout=torch.argmax(self.inference, dim=1)
        return self.inferenceout
    
    
    def loss(self,target):
        self.layers['softmax']=softmaxwithloss(self.inference,target)
        self.layers['softmax'].forward()
        self.layers['softmax'].loss()
        #print("loss: ", self.layers['softmax'].loss)
        return self.layers['softmax'].loss_value
    
    def backward(self):
        self.grad={}
        self.grad['beforesoft'] = self.layers['softmax'].backward()
        grads,self.grad['w2'],self.grad['b2'] = self.layers['affine2'].backward(self.grad['beforesoft'] )
        grads = self.layers['ReLU'].backward(grads)
        grads,self.grad['w1'],self.grad['b1'] = self.layers['affine1'].backward(grads)

        self.update()
        
    def update(self):
        # 更新权重的逻辑
        lr=0.001
        self.params['weight1'] -= lr * self.grad['w1']
        self.params['weight2'] -= lr * self.grad['w2']
        self.params['bias1'] -= lr * self.grad['b1']
        self.params['bias2'] -= lr * self.grad['b2']

    def inferences(self, x):
        x=self.forward(x)
        return self.inferenceout


# 设置数据处理流程
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的标准化参数
])

# 下载并加载训练数据集
train_dataset = datasets.MNIST(
    root='./data',          # 数据集存储路径
    train=True,             # 加载训练集
    download=True,          # 如果本地没有则下载
    transform=transform     # 应用定义的数据转换
)

# 下载并加载测试数据集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,            # 加载测试集
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,          # 每批加载64个样本
    shuffle=False            # 打乱训练数据顺序
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,        # 测试时使用更大的批次
    shuffle=True           # 测试数据不需要打乱
)

# model=twowisenetwork(,,)
# 测试数据加载
def inspect_data():
    # 获取一个批次的数据
    images, labels = next(iter(train_loader))

    


    print(f"数据集类型: {type(train_dataset)}")
    print(f"数据加载器类型: {type(train_loader)}")
    print(f"图像张量形状: {images.shape}")  # [batch, channels, height, width]
    print(f"标签形状: {labels.shape}")
    print(f"图像数据范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"标签示例: {labels[:10].tolist()}")

    # plt.imshow(images.squeeze(), cmap='gray') 
    # plt.title(f"Label: {labels}")
    
    # # 可视化样
    # plt.figure(figsize=(10, 5))
    # for i in range(12):
    #     plt.subplot(3, 4, i+1)
    #     plt.imshow(images[i].squeeze(), cmap='gray')  # 移除通道维度并显示
    #     plt.title(f"Label: {labels[i]}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('mnist_samples.png', dpi=120)
    # plt.show()
t=200
model=twowisenetwork(784, 1024, 10)  # MNIST图像大小为28x28=784，隐藏层128个神经元，输出层10个类别
loss=torch.zeros(t)  # 用于存储每个epoch的损失值
for epoch in range(t):
    images,labels=next(iter(train_loader))
    #标签处理
    labels = labels.long().view(-1)  # 保证是一维long类型
    labels = torch.nn.functional.one_hot(labels.squeeze(), num_classes=10).float()
    #print(labels)

    #图像数据一维化
    images=images.squeeze()
    images = images.view(64,784,1)
    inference=model.forward(images)
    #print(f"Inference Output: {inference}")
    loss[epoch]=model.loss(labels)
    print(f"Epoch {epoch+1}, Loss: {loss[epoch]}")


    model.backward()
    model.update()

pltx=range(1, 1+t)
plty=loss.numpy()
plt.plot(pltx, plty, marker='o')
plt.show()
##########################################################################
def visualize_samples(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')  # 移除通道维度并显示
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=120)
    plt.show()
##########################################################################
images,labels=next(iter(train_loader))
visualize_samples(images,labels)
    #标签处理
labels = labels.long().view(-1)  # 保证是一维long类型
labels = torch.nn.functional.one_hot(labels.squeeze(), num_classes=10).float()
    #print(labels)

    #图像数据一维化
images=images.squeeze()
images = images.view(64,784)
inferenceout=model.inferences(images[0:10,:])
print("**************************************************")
print(f"Inference Output: {inferenceout}")
print(f"Labels: {labels[0:10]}")
print("**************************************************")
loss[epoch]=model.loss(labels[0:10])
print(f"Epoch {epoch+1}, Loss: {loss[epoch]}")










    
    
 




# 这里的代码是一个简单的神经网络实现，包含了前向传播、损失计算和反向传播的逻辑。
# 该网络包含两个全连接层（affine），一个ReLU激活函数和一个softmax损失层。




# 执行检查
if __name__ == "__main__":

    inspect_data