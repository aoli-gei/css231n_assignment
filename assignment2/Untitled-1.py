loss,dscores=softmax_loss(scores,y)
for i in range(self.num_layers):
    W=self.params['W'+str(i+1)]
    loss+=0.5*self.reg*np.sum(W**2)

dout,dW,db=affine_backward(dscores,caches[self.num_layers-1])
dW+=self.reg*self.params['W'+str(self.num_layers)]

grads['W'+str(self.num_layers)]=dW
grads['b'+str(self.num_layers)]=db

for i in range(self.num_layers-2,-1,-1):
    if self.use_dropout:
        dout=dropout_backward(dout,dropout_caches[i])

    if self.normalization == None:
        dout,dW,db=affine_relu_backward(dout,caches[i])
        dW+=self.reg*self.params['W'+str(i+1)]
        grads['W'+str(i+1)]=dW
        grads['b'+str(i+1)]=db
    elif self.normalization == "batchnorm":
        dout,dW,db,dgamma,dbeta = affine_bn_relu_backward(dout,caches[i])
        dW+=self.reg*self.params['W'+str(i+1)]
        grads['W'+str(i+1)]=dW
        grads['b'+str(i+1)]=db
        grads['gamma'+str(i+1)]=dgamma
        grads['beta'+str(i+1)]=dbeta

    elif self.normalization=="layernorm":
        dout,dW,db,dgamma,dbeta=affine_ln_relu_backward(dout,caches[i])
        dW+=self.reg*self.params['W'+str(i+1)]
        grads['W'+str(i+1)]=dW
        grads['b'+str(i+1)]=db
        grads['gamma'+str(i+1)]=dgamma
        grads['beta'+str(i+1)]=dbeta