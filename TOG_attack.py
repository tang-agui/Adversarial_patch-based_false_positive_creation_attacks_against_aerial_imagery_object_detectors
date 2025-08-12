'''
实现TOG的vanishing attack
从攻击形式上来看，本质上仍然是一种类FGSM/PGD攻击
'''
'''
暂缓
'''

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TOG_Attack(object):
    '''
    默认参数也是TOG中给出
    '''
    def __init__(self, model, eps = 0.031, alpha = 2.0/255, steps = 10, targeted = False):
        super(TOG_Attack,self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.model = model
        self.sign = 1
        if targeted:
            self.sign = -1

    # def forward(self,images, labels):
    def calculate(self, images, labels):
        
        delta = torch.FloatTensor(images.size()).uniform_(-self.eps, self.eps)
        delta = delta.to(device)
        images = images.to(device)
        labels = labels.to(device)
        #   loss = nn.CrossEntropyLoss()

        adv_images_PGD = images.clone().detach()
        adv_images_PGD.requires_grad = True
        adv_images_PGD = torch.clamp(adv_images_PGD + delta, min = 0., max = 1.)

        for i in range(self.steps):
            # adv_images_PGD.requires_grad = True
            outputs = self.model(adv_images_PGD)
            self.model.zero_grad()
            cost = self.sign * loss(outputs,labels).to(device)
            cost.backward()

            adv_images_PGD = adv_images_PGD + self.alpha * adv_images_PGD.grad.sign()     ## 真实梯度求解
            delta = torch.clamp(adv_images_PGD - images, min = -self.eps, max = self.eps)
            adv_images_PGD = torch.clamp(images + delta, min = 0, max = 1).detach()

        return adv_images_PGD
    
    



'''
https://github.com/git-disl/TOG/blob/dc504b8ee773c78dfbf6918657755767ba597561/tog/attacks.py
def tog_fabrication(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_fabrication_gradient(x_adv)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv
'''