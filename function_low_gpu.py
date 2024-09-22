import torch
from torch import nn
import string
import random
import os

# thay thành false khi infer
is_training = False

def generate_tensor_file_name():
    while True:
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k = 10))
        
        if random_string not in os.listdir('computational_graph'):
            return random_string

#module.net có thể là module hoặc func
class One_Input_Call(torch.autograd.Function):
    def forward(context, module, input_tensor):
        context.module = module.net
        context.name = os.path.join("computational_graph", generate_tensor_file_name())
        print("forward : " + context.name[-10:])
        print(module.net)
        if is_training:
            torch.save(input_tensor, context.name)
        torch.cuda.empty_cache()
        return module.net(input_tensor)
        
    def backward(context, output_gradient):
        print("backward : " + context.name[-10:])
        print("Current GPU Usage : " + str(torch.cuda.memory_allocated(0) / 1024 ** 3))

        input_tensor = torch.load(context.name, weights_only = True)
        os.remove(context.name)
        with torch.enable_grad():
            output_tensor = context.module(input_tensor)
            output_tensor.backward(output_gradient)

        torch.cuda.empty_cache()
        return None, input_tensor.grad

class Two_Input_Call(torch.autograd.Function):
    def forward(context, module, input_tensor_1, input_tensor_2):
        if is_training:
            context.module = module.net
            context.name_1 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_1, context.name_1)
            context.name_2 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_2, context.name_2)
            print("forward : " + context.name_1[-10:] + " " + context.name_2[-10:])
        print(module.net)
        torch.cuda.empty_cache()
        return module.net(input_tensor_1, input_tensor_2)
        
    def backward(context, output_gradient):
        print("backward : " + context.name_1[-10:] + " " + context.name_2[-10:])
        print("Current GPU Usage : " + str(torch.cuda.memory_allocated(0) / 1024 ** 3))
        input_tensor_1 = torch.load(context.name_1, weights_only = True)
        input_tensor_2 = torch.load(context.name_2, weights_only = True)
        os.remove(context.name_1)
        os.remove(context.name_2)
        with torch.enable_grad():
            output_tensor = context.module(input_tensor_1, input_tensor_2)
            output_tensor.backward(output_gradient)
        torch.cuda.empty_cache()
        return None, input_tensor_1.grad, input_tensor_2.grad

class Three_Input_Call(torch.autograd.Function):
    def forward(context, module, input_tensor_1, input_tensor_2, input_tensor_3):
        if is_training:
            context.module = module.net
            context.name_1 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_1, context.name_1)
            context.name_2 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_2, context.name_2)
            context.name_3 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_3, context.name_3)
            print("forward : " + context.name_1[-10:] + " " + context.name_2[-10:] + " " + context.name_3[-10:])

        print(module.net)
        torch.cuda.empty_cache()
        return module.net(input_tensor_1, input_tensor_2, input_tensor_3)[0]
        
    def backward(context, output_gradient):
        print("backward : " + context.name_1[-10:] + " " + context.name_2[-10:] + " " + context.name_3[-10:])
        print("Current GPU Usage : " + str(torch.cuda.memory_allocated(0) / 1024 ** 3))
        input_tensor_1 = torch.load(context.name_1, weights_only = True)
        input_tensor_2 = torch.load(context.name_2, weights_only = True)
        input_tensor_3 = torch.load(context.name_3, weights_only = True)
        os.remove(context.name_1)
        os.remove(context.name_2)
        os.remove(context.name_3)
        with torch.enable_grad():
            output_tensor = context.module(input_tensor_1, input_tensor_2, input_tensor_3)[0]
            output_tensor.backward(output_gradient)
        torch.cuda.empty_cache()
        return None, input_tensor_1.grad, input_tensor_2.grad, input_tensor_3.grad

class Four_Input_Call(torch.autograd.Function):
    def forward(context, module, input_tensor_1, input_tensor_2, input_tensor_3, input_tensor_4):
        if is_training:
            context.module = module.net
            context.name_1 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_1, context.name_1)
            context.name_2 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_2, context.name_2)
            context.name_3 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_3, context.name_3)
            context.name_4 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_4, context.name_4)
            print(
                "forward : " + 
                context.name_1[-10:] + " " + 
                context.name_2[-10:] + " " + 
                context.name_3[-10:] + " " + 
                context.name_4[-10:]
            )
        print(module.net)
        torch.cuda.empty_cache()
        return module.net(input_tensor_1, input_tensor_2, input_tensor_3, attn_mask = input_tensor_4)[0]
        
    def backward(context, output_gradient):
        print(
            "backward : " + 
            context.name_1[-10:] + " " + 
            context.name_2[-10:] + " " + 
            context.name_3[-10:] + " " + 
            context.name_4[-10:]
        )
        print("Current GPU Usage : " + str(torch.cuda.memory_allocated(0) / 1024 ** 3))
        input_tensor_1 = torch.load(context.name_1, weights_only = True)
        input_tensor_2 = torch.load(context.name_2, weights_only = True)
        input_tensor_3 = torch.load(context.name_3, weights_only = True)
        input_tensor_4 = torch.load(context.name_4, weights_only = True)
        os.remove(context.name_1)
        os.remove(context.name_2)
        os.remove(context.name_3)
        os.remove(context.name_4)
        with torch.enable_grad():
            output_tensor = context.module(input_tensor_1, input_tensor_2, input_tensor_3, attn_mask = input_tensor_4)[0]
            output_tensor.backward(output_gradient)
        torch.cuda.empty_cache()
        return None, input_tensor_1.grad, input_tensor_2.grad, input_tensor_3.grad, input_tensor_4.grad

def one_input_forward(module, x):
    bearer = torch.tensor([], requires_grad = True, device = "cuda" if torch.cuda.is_available() else "cpu")
    bearer.net = module
    return One_Input_Call.apply(bearer, x)

def two_input_forward(module, x, y):
    bearer = torch.tensor([], requires_grad = True, device = "cuda" if torch.cuda.is_available() else "cpu")
    bearer.net = module
    return Two_Input_Call.apply(bearer, x, y)

def three_input_forward(module, x, y, z):
    bearer = torch.tensor([], requires_grad = True, device = "cuda" if torch.cuda.is_available() else "cpu")
    bearer.net = module
    return Three_Input_Call.apply(bearer, x, y, z)

def four_input_forward(module, x, y, z, w):
    bearer = torch.tensor([], requires_grad = True, device = "cuda" if torch.cuda.is_available() else "cpu")
    bearer.net = module
    return Four_Input_Call.apply(bearer, x, y, z, w)

class Modified_Multiply(torch.autograd.Function):
    def forward(context, input_tensor_1, input_tensor_2):
        if is_training:
            context.name_1 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_1, context.name_1)
            context.name_2 = os.path.join("computational_graph", generate_tensor_file_name())
            torch.save(input_tensor_2, context.name_2)
            print("forward : " + context.name_1[-10:] + " " + context.name_2[-10:])
        print("Modified Multiply")
        torch.cuda.empty_cache()
        return input_tensor_1 * input_tensor_2
        
    def backward(context, output_gradient):
        print("backward : " + context.name_1[-10:] + " " + context.name_2[-10:])
        input_tensor_1 = torch.load(context.name_1, weights_only = True)
        input_tensor_2 = torch.load(context.name_2, weights_only = True)
        os.remove(context.name_1)
        os.remove(context.name_2)
        torch.cuda.empty_cache()
        return output_gradient * input_tensor_2, output_gradient * input_tensor_1



