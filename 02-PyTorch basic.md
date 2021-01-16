# The Foundation of PyTorch Deep Learning
# PyTorch深度学习基础

## Learning experience
## 学习心得

I didn't have a foundation for deep learning before I learned PyTorch, and no one has experience of it for more than two years (PyTorch released in 2017), so we all study together.  
由于我也才学PyTorch，之前也没有基础，再有基础也不会超过两年（2017发布），所以大家都是一起学习。

That said, those acquainted with or open rather than skeptical to deep learning still have an edge. The point from the previous article here is that those who have not studied statistics or machine learning could go forward with burdens discarded. Personally, it is not recommended to read books on traditional machine learning method such as watermelon books first, just start with deep learning, and it's not late looking through these books for inspiration unitl it is difficult for you to break through.  
话虽如此，但是懂深度学习套路的，持开放而非怀疑态度的人还是很占优势的。引用上一篇的观点就是没学过统计、机器学习的就没有包袱，轻装上路。不推荐先阅读西瓜书等传统方法类书籍，先从深度学习入手，后期因为难以突破再翻看传统方法寻找灵感也来得及。


以工作为目的则要反其道行之，即从机器学习基础抓起。不能只会调库函数，既要有理论，又要会代码，推荐《机器学习实战》。最近实现了一下多项式回归，对于模拟的数据在一次函数上正常，二次及以上函数不收敛。花了大量时间检查梯度的更新过程，最后去找机器学习的注意事项，才想到可能的原因，没有归一化的数据导致梯度爆炸。发现很多介绍理论对代码不够注重的博客都忽略了对数据的归一化。

[Mofan Python](https://mofanpy.com/tutorials/machine-learning/torch/) is easy to follow and more practical than the [offical tutorials](https://pytorch.org/tutorials/). But it's still too simple for a real project.


Here I recommend [`Dive into Deep Learning` (PyTorch)](https://tangshusen.me/Dive-into-DL-PyTorch), and there are many other implemetations with different architectures, we can read them in contrast way.  
书籍只推荐[《动手学深度学习》(PyTorch版)](https://tangshusen.me/Dive-into-DL-PyTorch)，虽然有点挖墙脚，但是学有余力可以与MXNet对照，同时掌握两个框架。

As for my opinion on Tensorflow, I tried to learn the 1.x version, which has many functions and successfully dissuaded me. Although it's much more powerful than Pytorch, my suggestion is that don't try it first for your limited energy.  
至于我对Tensorflow的看法：曾经试着学习1.x版本，功能繁多，成功将我劝退。尽管确实要比PyTorch强大，然而精力有限，还是先别试了。


## 示例
## Examples

The following is an example of convolution  
以下为卷积示例

![convolution](images/conv1.gif)
```python
>>> import torch
>>> import torch.nn.functional as F
>>> data=torch.Tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]).unsqueeze_(0).unsqueeze_(0)
>>> data
tensor([[[[1., 1., 1., 0., 0.],
          [0., 1., 1., 1., 0.],
          [0., 0., 1., 1., 1.],
          [0., 0., 1., 1., 0.],
          [0., 1., 1., 0., 0.]]]])
>>> kernel=torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).unsqueeze_(0).unsqueeze_(0)
>>> kernel
tensor([[[[1., 0., 1.],
          [0., 1., 0.],
          [1., 0., 1.]]]])
>>> F.conv2d(data, kernel)
tensor([[[[4., 3., 4.],
          [2., 4., 3.],
          [2., 3., 4.]]]])
```


![convolution](images/conv2.gif)
```python
>>> import torch
>>> import torch.nn.functional as F
>>> data=torch.Tensor([
    [[0, 1, 1, 2, 2], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 2, 0, 1, 0]],
    [[1, 1, 1, 2, 0], [0, 2, 1, 1, 2], [1, 2, 0, 0, 2], [0, 2, 1, 2, 1], [2, 0, 1, 2, 0]],
    [[2, 0, 2, 0, 2], [0, 0, 1, 2, 1], [1, 0, 2, 2, 1], [2, 0, 2, 0, 0], [0, 0, 1, 1, 2]],
]).unsqueeze_(0)
>>> kernel=torch.Tensor([
    [[[1, 1, -1], [-1, 0, 1], [-1, -1, 0]],
     [[-1, 0, -1], [0, 0, -1], [1, -1, 0]],
     [[0, 1, 0], [1, 0, 1], [0, -1, 1]],],
    [[[-1, -1, 0], [-1, 1, 0], [-1, 1, 0]],
     [[1, -1, 0], [-1, 0, -1], [-1, 0, 0]],
     [[-1, 0, 1], [1, 0, 1], [0, -1, 0]],],
])
>>> F.conv2d(data, kernel, stride=2, padding=1,bias=torch.Tensor([1,0]))
tensor([[[[ 1.,  0., -3.],
          [-6.,  1.,  1.],
          [ 4., -3.,  1.]],

         [[-1., -6., -4.],
          [-2., -3., -4.],
          [-1., -3., -3.]]]])
```

该怎么理解呢？
data通过unsqueeze_(0)将大小为3的数据包裹成一个batch，内部的数据可以看成 $5 \times 5$ 的图片；
kernel则直接将2个卷积核包裹为1个batch，意思是最后通过卷积操作获得两张特征图。

上述过程对应
```
nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1, bias=True)
```
网络将自动更新kernel的数据，只需我们说明每个数据输出两个特征图，卷积核尺寸 $3 \times 3$，步长2，填充1，偏置bias不为零，即可。


>[CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv#torch.nn.Conv2d)  
>`CLASS torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, dilation: Union[T, Tuple[T, T]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')`  
>[source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d)


```python
>>> import torch
>>> import torch.nn as nn
>>> conv=nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1, bias=True)
>>> conv
Conv2d(3, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
>>> kernel=torch.Tensor([
    [[[1, 1, -1], [-1, 0, 1], [-1, -1, 0]],
     [[-1, 0, -1], [0, 0, -1], [1, -1, 0]],
     [[0, 1, 0], [1, 0, 1], [0, -1, 1]],],
    [[[-1, -1, 0], [-1, 1, 0], [-1, 1, 0]],
     [[1, -1, 0], [-1, 0, -1], [-1, 0, 0]],
     [[-1, 0, 1], [1, 0, 1], [0, -1, 0]],],
])
>>> data=torch.Tensor([
    [[0, 1, 1, 2, 2], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 2, 0, 1, 0]],
    [[1, 1, 1, 2, 0], [0, 2, 1, 1, 2], [1, 2, 0, 0, 2], [0, 2, 1, 2, 1], [2, 0, 1, 2, 0]],
    [[2, 0, 2, 0, 2], [0, 0, 1, 2, 1], [1, 0, 2, 2, 1], [2, 0, 2, 0, 0], [0, 0, 1, 1, 2]],
]).unsqueeze_(0)
>>> conv(data) #random initialization
tensor([[[[-0.2415, -0.6459, -0.8011],
          [ 0.3015, -1.3054, -0.0484],
          [-0.1819, -0.2977, -0.5712]],

         [[-0.0173,  0.3147,  0.0089],
          [ 0.1098,  0.3265, -0.0871],
          [ 0.2995,  0.2101,  0.4130]]]], grad_fn=<ThnnConv2DBackward>)
>>> conv.weight = nn.Parameter(kernel)
>>> conv.bias = nn.Parameter(torch.Tensor(2).zero_())
>>> conv.bias
Parameter containing:
tensor([0., 0.], requires_grad=True)
>>> conv(data)
tensor([[[[ 0., -1., -4.],
          [-7.,  0.,  0.],
          [ 3., -4.,  0.]],

         [[-1., -6., -4.],
          [-2., -3., -4.],
          [-1., -3., -3.]]]], grad_fn=<ThnnConv2DBackward>)
```

这样就算对一个PyTorch的nn模块有了基本的认识了。若只想把这些nn模块当黑盒来用，不想考虑细节的话，上面就权当是初始化的一个简介好了。我的目标则是能从零开始构建出常用模块，所以`grad_fn`的实现以及计算优化会是我的关注点。下面是一个小技巧，说明为何在PyTorch中不用显式调用forward。


```python
>>> class LayerBase:
    def __init__(self, *args, **kwargs):
        """make the class name "LayerBase" like a function"""
        print("called: __init__", "args:", args, "kwargs", kwargs)
        self.name = args[0] if args else "unknown"
    def __call__(self, *args, **kwargs):
        """make an instance of the class "LayerBase" like a function"""
        print("called: __call__", self.name, "args:", args, "kwargs", kwargs)
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        print("called: forward", self.name, "args:", args, "kwargs", kwargs)
        return str(args) + str(kwargs)
>>> n = LayerBase(1, a=2)
called: __init__ args: (1,) kwargs {'a': 2}
>>> n(c=(3,1), d='linear')
called: __call__ 1 args: () kwargs {'c': (3, 1), 'd': 'linear'}
called: forward 1 args: () kwargs {'c': (3, 1), 'd': 'linear'}
"(){'c': (3, 1), 'd': 'linear'}"
>>> n(["as", 12], 12, c='linear')
called: __call__ 1 args: (['as', 12], 12) kwargs {'c': 'linear'}
called: forward 1 args: (['as', 12], 12) kwargs {'c': 'linear'}
"(['as', 12], 12){'c': 'linear'}"

# so now we can derive
>>> class Layers(LayerBase):
    def __init__(self, *args, **kwargs):
        """make the class name "LayerBase" like a function"""
        super().__init__(*args, **kwargs)
        self.layer1 = LayerBase('name1')
        self.layer2 = LayerBase('name2')
    def forward(self, input):
        print("called: forward", self.name, "input:", input)
        x = self.layer1(input)
        x = self.layer2(x)
        return x
>>> m = Layers(1, 's')
called: __init__ args: (1, 's') kwargs {}
called: __init__ args: ('name1',) kwargs {}
called: __init__ args: ('name2',) kwargs {}
>>> m('qwerty')
called: __call__ 1 args: ('qwerty',) kwargs {}
called: forward 1 input: qwerty
called: __call__ name1 args: ('qwerty',) kwargs {}
called: forward name1 args: ('qwerty',) kwargs {}
called: __call__ name2 args: ("('qwerty',){}",) kwargs {}
called: forward name2 args: ("('qwerty',){}",) kwargs {}
'("(\'qwerty\',){}",){}'
```

With the above two static parts, and aotomatic derivative(gradient, not easy for an example), we can get a normal nn module in PyTorch.

以后我将补上 backward 和 gradient 的具体实现。

### Ref

- https://www.zhihu.com/question/375537442/answer/1352795939

- https://www.jianshu.com/p/f97791393439
